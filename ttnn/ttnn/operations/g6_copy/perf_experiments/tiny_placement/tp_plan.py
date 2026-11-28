"""tiny_placement: host-side placement / core-count plans for g6_copy, built on the op's own
_Plan/_build (the op's kernels are used unchanged). Nothing here edits the op.

A placement is a list of (core, first, count) where first/count index either the natural page
order (bank_major=False, the op's default walk) or the bank-major enumeration (bank_major=True,
the op's BANK_AFFINITY walk: position j = all pages of bank 0, then bank 1, ...).
"""
import ttnn
import ttnn.operations.g6_copy.g6_copy_program_descriptor as PD

NUM_BANKS = 8


def bank_major_ranges(tensor_pages, num_banks=NUM_BANKS):
    """[(bank, j_first, n_pages)] of the bank-major enumeration (q+1 pages for the first r banks)."""
    q, r = divmod(tensor_pages, num_banks)
    out, j = [], 0
    for b in range(num_banks):
        n = q + (1 if b < r else 0)
        if n:
            out.append((b, j, n))
        j += n
    return out


def make_plan(work, tensor_pages, page_bytes, grid):
    """_Plan for an explicit work list [(CoreCoord, first, count, rotation)], one_block or streamed
    exactly as the op's _plan picks it."""
    cores = [w[0] for w in work]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    work = tuple(PD._CoreWork(c, f, n, r) for c, f, n, r in work)
    core_pages_max = max(w.num_pages for w in work)
    block_target = max(1, PD.BLOCK_BYTES // page_bytes)
    if core_pages_max <= block_target:
        plan, block_pages, depth = "one_block", core_pages_max, 1
    else:
        plan, block_pages = "streamed", block_target
        depth = min(PD.STREAM_DEPTH_BLOCKS, PD.L1_BUDGET_BYTES // (block_pages * page_bytes))
    return PD._Plan(
        plan=plan,
        grid=(grid.x, grid.y),
        all_cores=crs,
        work=work,
        tensor_pages=tensor_pages,
        page_bytes=page_bytes,
        block_pages=block_pages,
        cb_depth_blocks=depth,
        read_trids=1,
    )


def build(inp, out, plan, bank_major, reader_noc=ttnn.NOC.NOC_0, writer_noc=ttnn.NOC.NOC_1):
    """The op's _build (same CB, same kernels, same args) with BANK_AFFINITY and each kernel's NoC set
    explicitly. Reader stays on NCRISC (RISCV_1), writer on BRISC (RISCV_0)."""
    rc = ttnn.DataMovementConfigDescriptor()
    rc.processor, rc.noc = ttnn.DataMovementProcessor.RISCV_1, reader_noc
    wc = ttnn.DataMovementConfigDescriptor()
    wc.processor, wc.noc = ttnn.DataMovementProcessor.RISCV_0, writer_noc
    old = PD.BANK_AFFINITY
    PD.BANK_AFFINITY = bool(bank_major)
    try:
        cb = ttnn.CBDescriptor(
            total_size=plan.cb_pages * plan.page_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=PD.CB_PAGE_STAGING, data_format=inp.dtype, page_size=plan.page_bytes
                )
            ],
        )
        reader = PD._kernel(PD.READER_KERNEL, inp, plan, rc)
        writer = PD._kernel(PD.WRITER_KERNEL, out, plan, wc)
    finally:
        PD.BANK_AFFINITY = old
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb])


def optimal_cores(device):
    """Per bank: the logical worker tt-metal reports nearest to that bank's NoC0 / NoC1 endpoint."""
    o0 = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    o1 = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_1)
    return [(c.x, c.y) for c in o0], [(c.x, c.y) for c in o1]


def _take_core(row, used, grid, bank_rows=()):
    """First free core in `row` (lowest x); when the row is full, the rows no bank is aligned with
    (nearest first), then any row."""
    spare = sorted((y for y in range(grid.y) if y not in bank_rows), key=lambda y: abs(y - row))
    rest = sorted((y for y in range(grid.y) if y in bank_rows and y != row), key=lambda y: abs(y - row))
    for y in [row] + spare + rest:
        for x in range(grid.x):
            if (x, y) not in used:
                used.add((x, y))
                return ttnn.CoreCoord(x, y)
    raise RuntimeError("grid exhausted")


def _split(n, c):
    q, r = divmod(n, c)
    out, f = [], 0
    for i in range(c):
        k = q + (1 if i < r else 0)
        out.append((f, k))
        f += k
    return out


def placed_work(tensor_pages, grid, bank_rows, cores_per_bank, num_banks=NUM_BANKS):
    """Candidate: bank-major split, each bank's pages over min(cores_per_bank, pages) cores, every core
    on the worker row of its bank's NoC endpoint (bank_rows[b]). Returns [(core, j_first, n, rot)]."""
    used, work = set(), []
    for b, j0, nb in bank_major_ranges(tensor_pages, num_banks):
        for f, c in _split(nb, min(cores_per_bank, nb)):
            work.append((_take_core(bank_rows[b], used, grid, set(bank_rows)), j0 + f, c, len(work)))
    return work


def placed_work_v(tensor_pages, grid, bank_rows, cores_per_bank, mode, num_banks=NUM_BANKS):
    """Variants of the row budget. 'fill' = placed_work. 'balanced': banks sharing a row split it
    evenly, the rest of their cores go to spare rows. 'aligned': a bank only ever uses its share of its
    own row (fewer cores for banks that share a row, never an unaligned core)."""
    if mode == "fill":
        return placed_work(tensor_pages, grid, bank_rows, cores_per_bank, num_banks)
    share = {}
    for b in range(num_banks):
        sharers = [c for c in range(num_banks) if bank_rows[c] == bank_rows[b]]
        q, r = divmod(grid.x, len(sharers))
        share[b] = q + (1 if sharers.index(b) < r else 0)
    used, work = set(), []
    row_next = {}
    spare_rows = [y for y in range(grid.y) if y not in set(bank_rows)]
    spare = [(x, y) for y in spare_rows for x in range(grid.x)]
    for b, j0, nb in bank_major_ranges(tensor_pages, num_banks):
        k = min(cores_per_bank, nb)
        if mode == "aligned":
            k = min(k, share[b])
        cores = []
        row = bank_rows[b]
        for i in range(k):
            if i < share[b]:
                x = row_next.get(row, 0)
                row_next[row] = x + 1
                cores.append(ttnn.CoreCoord(x, row))
            else:
                x, y = spare.pop(0)
                cores.append(ttnn.CoreCoord(x, y))
        for (f, c), core in zip(_split(nb, k), cores):
            work.append((core, j0 + f, c, len(work)))
    return work
