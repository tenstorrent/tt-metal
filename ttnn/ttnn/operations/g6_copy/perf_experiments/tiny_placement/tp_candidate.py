"""tiny_placement candidate, written as the host code that would replace the op's work split.

Rule (one path, every shape):
  * Pages are enumerated bank-major (the op's BANK_AFFINITY walk): each bank's pages form one range.
  * Each bank's range is split over k = min(ceil(pages / num_banks), grid.x) cores, so at most one
    page-run per core until a bank has more than grid.x pages.
  * Cores of bank b sit on the worker row of b's NoC0 DRAM endpoint (row of
    get_optimal_dram_bank_to_logical_worker_assignment(device, NOC_0)[b]) while that row has a free core;
    once it is full (banks 0 and 4 share a row on p150b) the bank's remaining cores take the rows no
    bank maps to (nearest first), never another bank's row. Column does not matter (NoC is a unidirectional torus: a round trip pays the full
    X ring whatever the column, and the full Y ring unless the core shares the endpoint's row).
  * Reader (NCRISC) AND writer (BRISC) both on NoC0: every bank's NoC0 endpoint lands on a worker row,
    banks 0/4's NoC1 endpoints do not (physical row 1), and a row-aligned core's traffic never leaves
    its row.
"""
import ttnn
import ttnn.operations.g6_copy.g6_copy_program_descriptor as PD


def bank_rows_noc0(device):
    return [c.y for c in ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)]


def _split(n, c):
    q, r = divmod(n, c)
    out, f = [], 0
    for i in range(c):
        k = q + (1 if i < r else 0)
        out.append((f, k))
        f += k
    return out


def plan(grid, tensor_pages, page_bytes, bank_rows):
    num_banks = len(bank_rows)
    k_max = min(-(-tensor_pages // num_banks), grid.x)
    q, r = divmod(tensor_pages, num_banks)
    aligned = set(bank_rows)
    used, work = set(), []

    def free_in(rows):
        for y in rows:
            for x in range(grid.x):
                if (x, y) not in used:
                    return x, y
        return None

    j0 = 0
    for b in range(num_banks):
        nb = q + (1 if b < r else 0)
        if nb == 0:
            continue
        row = bank_rows[b]
        spare = sorted((y for y in range(grid.y) if y not in aligned), key=lambda y: abs(y - row))
        anyrow = sorted(range(grid.y), key=lambda y: abs(y - row))
        for f, c in _split(nb, min(k_max, nb)):
            xy = free_in([row]) or free_in(spare) or free_in(anyrow)
            used.add(xy)
            work.append(PD._CoreWork(ttnn.CoreCoord(*xy), j0 + f, c, 0))
        j0 += nb
    assert j0 == tensor_pages

    crs = ttnn.CoreRangeSet([ttnn.CoreRange(w.core, w.core) for w in work])
    core_pages_max = max(w.num_pages for w in work)
    block_target = max(1, PD.BLOCK_BYTES // page_bytes)
    if core_pages_max <= block_target:
        kind, block_pages, depth = "one_block", core_pages_max, 1
    else:
        kind, block_pages = "streamed", block_target
        depth = min(PD.STREAM_DEPTH_BLOCKS, PD.L1_BUDGET_BYTES // (block_pages * page_bytes))
    return PD._Plan(
        plan=kind,
        grid=(grid.x, grid.y),
        all_cores=crs,
        work=tuple(work),
        tensor_pages=tensor_pages,
        page_bytes=page_bytes,
        block_pages=block_pages,
        cb_depth_blocks=depth,
        read_trids=1,
    )


def build(inp, out, p):
    rc = ttnn.DataMovementConfigDescriptor()
    rc.processor, rc.noc = ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0
    wc = ttnn.DataMovementConfigDescriptor()
    wc.processor, wc.noc = ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0
    old = PD.BANK_AFFINITY
    PD.BANK_AFFINITY = True  # bank-major walk (common RT arg 4)
    try:
        cb = ttnn.CBDescriptor(
            total_size=p.cb_pages * p.page_bytes,
            core_ranges=p.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=PD.CB_PAGE_STAGING, data_format=inp.dtype, page_size=p.page_bytes)
            ],
        )
        reader = PD._kernel(PD.READER_KERNEL, inp, p, rc)
        writer = PD._kernel(PD.WRITER_KERNEL, out, p, wc)
    finally:
        PD.BANK_AFFINITY = old
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb])


_CACHE = {}


def get_program_descriptor(input_tensor, output_tensor):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    key = (
        device.id(),
        (grid.x, grid.y),
        tuple(input_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.buffer_num_pages(),
        input_tensor.buffer_aligned_page_size(),
    )
    d = _CACHE.get(key)
    if d is None:
        p = plan(grid, input_tensor.buffer_num_pages(), input_tensor.buffer_aligned_page_size(), bank_rows_noc0(device))
        d = build(input_tensor, output_tensor, p)
        _CACHE[key] = d
    else:
        d.kernels[0].common_runtime_args[PD._CRTA_BUFFER_ADDRESS] = input_tensor.buffer_address()
        d.kernels[1].common_runtime_args[PD._CRTA_BUFFER_ADDRESS] = output_tensor.buffer_address()
    return d
