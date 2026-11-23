# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for wq_b_linear (N-split, x multicast resident, W streamed per worker).

Cores:
  * NUM_X_SENDERS x-senders (outside the receiver rectangle): each NCRISC reads its x rounds
    (block_m_tiles x block_k_tiles tiles each, X_READ_TRIDS rounds in flight) from DRAM into
    cb_x_resident; its BRISC multicasts each round (linked) to the same address on every worker and
    publishes it on the channel's Counter (ordered count write, no handshake).
  * num_workers workers (one rectangle): reader streams the worker's W columns per K-step,
    writer waits for each x round and pushes it to compute, compute runs one matmul_block over
    all K-steps, writer stores the 5 x block_n_tiles output block into the L1-interleaved y.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"  # this experiment dir's copy
TILE = 32

# ---- Blocking knobs (single source of truth; see op_design.md "Blocking Model") ----
BLOCK_N_TILES = 2  # output columns per worker; num_workers = tensor_n_tiles / BLOCK_N_TILES
# K tiles per K-step (matmul in0_block_k) = one x multicast round; num_k_blocks derived.
# Precision ceiling: with 16-bit DEST each K-step accumulates block_k_tiles products in DEST, and the
# DEST rounding drifts the output scale upward with that length (measured slope 1.0002 @2, 1.007 @8,
# 1.013 @16, 1.018 @24; PCC 0.99996 / 0.99993 / 0.99986 / 0.99979). Keep block_k_tiles <= 16 so the
# 0.9998 PCC gate holds (verification_report.md, precision baseline).
# Refinement 1 measured 8 -> 4: the last K-step's compute after the final W block halves, and with the
# ordered x signal a round costs no fixed ~1 us any more (kb 2 then measured slower).
# Perf 1: 4 -> 2. kb 2's loss came from the old compute's per-K-step interm drain (one exposed pack latency
# per K-step); with compute L1-accumulating straight into the output CB, kb 2 measured 18.61 / 18.66 us vs
# kb 4 19.05 / 19.33 us (x round 0 lands ~0.5 us earlier; every worker finishes together).
BLOCK_K_TILES = 2
# cb_w_block depth in K-steps (None = num_k_blocks, whole W slice resident). Refinement 2: 4 paces each
# worker's W stream to its compute, so early-served workers stop crowding the banks; the slowest worker
# now finishes with the median (measured 20.6 -> 20.2 us; depth 2/3/6 tie or lose). Frees 70 KB L1.
W_DEPTH_BLOCKS = 4
OUT_DEPTH_BLOCKS = 1  # one output block per core
MCAST_RECT_WIDTH_MAX = 8

# ---- x-delivery knobs (Refinement 1; numbers in changelog.md). The Phase 0 schedule is
# X_READ_TRIDS = 1, X_MCAST_LINKED = False, X_ORDERED_SIGNAL = False, X_SIGNAL_LAG = False,
# NUM_X_SENDERS = 1, W_WAIT_X0 = False. ----
X_READ_TRIDS = 2  # x rounds whose DRAM reads stay in flight on a sender (trid window, 1..15)
X_MCAST_LINKED = True  # one linked path reservation per x round instead of one per 16 KB burst
# Publish each round by multicasting its count into the receivers' Counter cell right behind the
# payload (ordered on the same path) instead of write barrier + atomic increment + atomic barrier.
X_ORDERED_SIGNAL = True
X_SIGNAL_LAG = True  # atomic signal only: publish round r-1 after issuing round r's payload
# x senders: each owns every NUM_X_SENDERS-th x round on its own Mcast channel (its own Counter).
# X_SENDER_NOCS[s] is sender s's multicast NoC (its BRISC); its x-reader (NCRISC) uses the other NoC.
# Sender 0 sits at the grid's bottom-right corner; a NoC1 sender 1 sits just above it, a NoC0
# sender 1 at the rectangle's top-left diagonal (NoC0 multicasts low -> high). 1..2 senders.
# Measured: with the ordered signal one NoC1 sender wins (a NoC0 sender slows the W reads that share
# NoC0; two NoC1 trees collide).
NUM_X_SENDERS = 1
X_SENDER_NOCS = (ttnn.NOC.NOC_1, ttnn.NOC.NOC_0)
X_MCAST_PREFIXES = ("x_mcast", "x_mcast_b")  # worker channel names, one per sender slot
W_WAIT_X0 = False  # hold each worker's W stream until x round 0 lands (measured slower; kept as a knob)

# ---- Tail / W-side knobs (Refinement 2; numbers in changelog.md). The Refinement 1 schedule is
# COLUMN_MAP = "bank_spread", W_COALESCE = False, OUT_SUBBLOCK_MAJOR = False (knob removed in Perf 1), Y_STORE_NOC_SPLIT = False,
# W_DEPTH_BLOCKS = None. ----
# Column map: "bank_spread" (worker j: block_n_tiles columns of one bank, residue j % num_banks) or
# "contiguous" (worker j: columns j*block_n_tiles .. +block_n_tiles-1, one per bank).
COLUMN_MAP = "bank_spread"
# One read per W k-row when the worker's columns sit in consecutive slots of one bank (bank_spread with
# tensor_n_tiles % num_banks == 0); silently off otherwise.
W_COALESCE = True
Y_STORE_NOC_SPLIT = True  # writer sends every other output tile on NoC0 (idle at the tail)

# ---- Ablation (perf measurement only; output is wrong when any is set). Stub a stage's payload, keep its sync.
# Members: "w_read" (worker DRAM W reads), "x_read" (sender DRAM x reads), "x_mcast" (x multicast payload; the
# count signal still goes out), "compute" (matmul LLK, helper SKIP_COMPUTE), "y_store" (y writes). ----
ABLATE = frozenset()

# ---- x_read_fast experiment knobs ----
X_READER = "base"  # "base" = the op's current x-reader (kernels/wq_b_linear_x_reader_base.cpp); "fast" = variant kernel
X_SPLIT_NOC = False  # fast: odd k tiles of each round read on the other NoC
X_FAST_ISSUE = False  # fast: precomputed bank addresses + one-packet stateful reads
X_PROBE_CHUNK = 0  # fast: >0 = perf probe, contiguous per-bank chunk reads (WRONG layout)
X_PROBE_UNTAGGED = False  # fast: perf probe, all rounds untagged + one barrier
X_PROBE_TAGGED = False  # with X_PROBE_UNTAGGED: tag each round with a trid (no waits)
X_HELPER = False  # fast: a second core (above the sender) reads the odd x rounds and writes them to the sender
X_EAGER = False  # fast: push each round as soon as its trid drains


def _ablate_defines(*stages):
    names = {
        "w_read": "ABLATE_W_READ",
        "x_read": "ABLATE_X_READ",
        "x_mcast": "ABLATE_X_MCAST",
        "compute": "SKIP_COMPUTE",
        "y_store": "ABLATE_Y_STORE",
    }
    ablate = set(ABLATE.split("+")) - {""} if isinstance(ABLATE, str) else set(ABLATE)  # "a+b" form for WQB_KNOBS
    assert ablate <= set(names), f"unknown ABLATE stage in {ABLATE}"
    return [(names[s], "1") for s in stages if s in ablate]


def _dest_tile_limit(fp32_dest_acc_en):
    """DEST_AUTO_LIMIT (half-sync): 8 tiles at 16-bit DEST, 4 at fp32 DEST."""
    return 4 if fp32_dest_acc_en else 8


# ---- CB indices (semantic names) ----
CB_X_RESIDENT = 0
CB_W_BLOCK = 1
CB_OUT_BLOCK = 2  # compute L1-accumulates every K-step straight into it (Perf 1; no separate interm CB)


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _choose_subblock(block_m, block_n, limit):
    """Largest h*w <= limit with h | block_m and w | block_n (ties -> wider w)."""
    best = (1, 1)
    for h in range(1, block_m + 1):
        if block_m % h:
            continue
        for w in range(1, block_n + 1):
            if block_n % w or h * w > limit:
                continue
            if h * w > best[0] * best[1] or (h * w == best[0] * best[1] and w > best[1]):
                best = (h, w)
    return best


def _bank_spread_columns(tensor_n_tiles, block_n_tiles, num_workers, num_banks):
    """Column map: worker j takes block_n_tiles columns from residue class j % num_banks."""
    residues = [[n for n in range(tensor_n_tiles) if n % num_banks == r] for r in range(num_banks)]
    nexts = [0] * num_banks
    cols = []
    for j in range(num_workers):
        r = j % num_banks
        take = residues[r][nexts[r] : nexts[r] + block_n_tiles]
        nexts[r] += len(take)
        cols.append(list(take))
    leftovers = [c for r in range(num_banks) for c in residues[r][nexts[r] :]]
    for j in range(num_workers):
        while len(cols[j]) < block_n_tiles:
            cols[j].append(leftovers.pop(0))
    flat = sorted(c for cs in cols for c in cs)
    assert flat == list(range(tensor_n_tiles)), "column map must assign each column exactly once"
    return cols


def _worker_geometry(device, num_workers):
    grid = device.compute_with_storage_grid_size()
    cap = min(MCAST_RECT_WIDTH_MAX, grid.x - 1)
    rect_w = max(d for d in range(1, cap + 1) if num_workers % d == 0)
    rect_h = num_workers // rect_w
    if rect_h > grid.y:
        raise RuntimeError(f"wq_b_linear: {num_workers} workers do not fit a {rect_w}x{rect_h} rect on {grid}")
    x0 = grid.x - 1 - rect_w
    y0 = grid.y - rect_h
    workers = [(x, y) for y in range(y0, y0 + rect_h) for x in range(x0, x0 + rect_w)]  # row-major
    rect_lo, rect_hi = ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + rect_w - 1, grid.y - 1)
    rect = ttnn.CoreRangeSet([ttnn.CoreRange(rect_lo, rect_hi)])
    senders = [(grid.x - 1, grid.y - 1)]
    if NUM_X_SENDERS > 1:
        senders.append((grid.x - 1, grid.y - 2) if X_SENDER_NOCS[1] == ttnn.NOC.NOC_1 else (x0 - 1, y0 - 1))
    for sx, sy in senders:
        assert 0 <= sx < grid.x and 0 <= sy < grid.y and not (x0 <= sx < x0 + rect_w and sy >= y0)
    return workers, rect, (rect_lo, rect_hi), senders


def create_program_descriptor(x, w, y, compute_config) -> ttnn.ProgramDescriptor:
    device = x.device()

    # ---- Geometry in tiles ----
    tensor_m_tiles = (x.shape[-2] + TILE - 1) // TILE
    tensor_k_tiles = x.shape[-1] // TILE
    tensor_n_tiles = w.shape[-1] // TILE

    block_m_tiles = tensor_m_tiles  # m not split (one band)
    block_k_tiles = BLOCK_K_TILES
    block_n_tiles = BLOCK_N_TILES
    assert tensor_k_tiles % block_k_tiles == 0, "tensor_k_tiles must divide by block_k_tiles"
    assert tensor_n_tiles % block_n_tiles == 0, "tensor_n_tiles must divide by block_n_tiles"
    num_k_blocks = tensor_k_tiles // block_k_tiles
    num_workers = tensor_n_tiles // block_n_tiles
    x_depth_rounds = num_k_blocks  # whole x resident (handshake-free multicast)
    w_depth_blocks = num_k_blocks if W_DEPTH_BLOCKS is None else W_DEPTH_BLOCKS

    fp32_dest_acc_en = bool(compute_config.fp32_dest_acc_en)
    sb_h, sb_w = _choose_subblock(block_m_tiles, block_n_tiles, _dest_tile_limit(fp32_dest_acc_en))
    in0_num_subblocks = block_m_tiles // sb_h
    in1_num_subblocks = block_n_tiles // sb_w

    assert 1 <= NUM_X_SENDERS <= len(X_MCAST_PREFIXES), "NUM_X_SENDERS out of range"
    workers, rect, (rect_lo, rect_hi), senders = _worker_geometry(device, num_workers)
    worker_crs = _crs(workers)
    all_crs = _crs(workers + senders)  # cb_x_resident: one address on every sender and worker
    grid = device.compute_with_storage_grid_size()
    helper = (grid.x - 1, grid.y - 2) if X_HELPER else None
    if helper is not None:
        assert NUM_X_SENDERS == 1 and helper not in workers
        all_crs = _crs(workers + senders + [helper])

    num_banks = device.dram_grid_size().x
    if COLUMN_MAP == "bank_spread":
        cols = _bank_spread_columns(tensor_n_tiles, block_n_tiles, num_workers, num_banks)
    elif COLUMN_MAP == "contiguous":
        cols = [list(range(j * block_n_tiles, (j + 1) * block_n_tiles)) for j in range(num_workers)]
    else:
        raise ValueError(f"wq_b_linear: unknown COLUMN_MAP {COLUMN_MAP!r}")
    # Coalescing needs page(k, c + i*num_banks) in consecutive slots of one bank: every row starts on the
    # same bank (tensor_n_tiles % num_banks == 0) and the worker's columns step by num_banks.
    w_coalesce = (
        W_COALESCE
        and tensor_n_tiles % num_banks == 0
        and all(cs == [cs[0] + i * num_banks for i in range(block_n_tiles)] for cs in cols)
    )

    # ---- Page sizes ----
    x_tile_bytes = x.buffer_page_size()
    w_tile_bytes = w.buffer_page_size()
    y_tile_bytes = y.buffer_page_size()
    x_round_tiles = block_m_tiles * block_k_tiles
    w_block_tiles = block_k_tiles * block_n_tiles
    out_block_tiles = block_m_tiles * block_n_tiles

    # ---- CBs. cb_x_resident first and on the union {workers, sender} so its address matches. ----
    cb_x_resident = ttnn.CBDescriptor(
        total_size=x_round_tiles * x_depth_rounds * x_tile_bytes,
        core_ranges=all_crs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_X_RESIDENT, data_format=x.dtype, page_size=x_tile_bytes)
        ],
    )
    cb_w_block = ttnn.CBDescriptor(
        total_size=w_block_tiles * w_depth_blocks * w_tile_bytes,
        core_ranges=worker_crs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_W_BLOCK, data_format=w.dtype, page_size=w_tile_bytes)
        ],
    )
    cb_out_block = ttnn.CBDescriptor(
        total_size=out_block_tiles * OUT_DEPTH_BLOCKS * y_tile_bytes,
        core_ranges=worker_crs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUT_BLOCK, data_format=y.dtype, page_size=y_tile_bytes)
        ],
    )

    # ---- x senders: x-reader (DRAM, one NoC) -> cb_x_resident -> multicaster (the other NoC) ----
    x_read_trids = min(X_READ_TRIDS, num_k_blocks)
    mc_lo = device.worker_core_from_logical_core(rect_lo)
    mc_hi = device.worker_core_from_logical_core(rect_hi)
    sender_kernels, x_reader_kernels = [], []
    for s_idx, (scx, scy) in enumerate(senders):
        mcast_noc = X_SENDER_NOCS[s_idx]
        read_noc = ttnn.NOC.NOC_0 if mcast_noc == ttnn.NOC.NOC_1 else ttnn.NOC.NOC_1
        round_sel = [s_idx, len(senders)]  # first_round, round_stride
        x_reader_ct = [CB_X_RESIDENT, block_m_tiles, block_k_tiles, num_k_blocks, tensor_k_tiles, x_read_trids]
        x_reader_ct.extend(round_sel)
        x_reader_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        x_reader_rt = ttnn.RuntimeArgs()
        x_reader_rt[scx][scy] = [x.buffer_address()]
        x_reader_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(
                    KERNEL_DIR / ("wq_b_linear_x_reader_base.cpp" if X_READER == "base" else "wq_b_linear_x_reader.cpp")
                ),
                core_ranges=_crs([(scx, scy)]),
                compile_time_args=x_reader_ct,
                runtime_args=x_reader_rt,
                defines=_ablate_defines("x_read")
                + [
                    (n, "1")
                    for n, on in (
                        ("X_SPLIT_NOC", X_SPLIT_NOC),
                        ("X_FAST_ISSUE", X_FAST_ISSUE),
                        ("X_EAGER", X_EAGER),
                        ("X_PROBE_UNTAGGED", X_PROBE_UNTAGGED),
                        ("X_PROBE_TAGGED", X_PROBE_TAGGED),
                    )
                    if on
                ]
                + [("X_NUM_BANKS", str(device.dram_grid_size().x))]
                + ([("X_PROBE_CHUNK", str(X_PROBE_CHUNK))] if X_PROBE_CHUNK else [])
                + ([("X_HELPER", "1"), ("X_HELPER_SEM", "XHSEM")] if X_HELPER else []),
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=read_noc),
            )
        )
        sender_ct = [CB_X_RESIDENT, block_m_tiles, block_k_tiles, num_k_blocks, int(X_MCAST_LINKED), int(X_SIGNAL_LAG)]
        sender_ct.extend(round_sel)
        # Channel s's Counter cell id: each no-handshake Counter channel takes one semaphore, allocated
        # in sender order from id 0 (asserted against the helper's allocation below).
        sender_ct.extend([int(X_ORDERED_SIGNAL), s_idx])
        # Receiver rectangle in NoC coordinates, in routing order for the multicast NoC (NoC1 runs
        # high -> low, so its start is the high corner).
        mc_start, mc_end = (mc_hi, mc_lo) if mcast_noc == ttnn.NOC.NOC_1 else (mc_lo, mc_hi)
        sender_rt = ttnn.RuntimeArgs()
        sender_rt[scx][scy] = [mc_start.x, mc_start.y, mc_end.x, mc_end.y, num_workers]
        sender_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "wq_b_linear_sender.cpp"),
                core_ranges=_crs([(scx, scy)]),
                compile_time_args=sender_ct,
                runtime_args=sender_rt,
                defines=_ablate_defines("x_mcast"),
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=mcast_noc),
            )
        )

    # ---- Worker reader (W columns per K-step), NCRISC / NoC0 ----
    reader_ct = [
        CB_W_BLOCK,
        block_k_tiles,
        block_n_tiles,
        num_k_blocks,
        tensor_n_tiles,
        int(W_WAIT_X0),
        int(w_coalesce),
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    reader_rt = ttnn.RuntimeArgs()
    for j, (cx, cy) in enumerate(workers):
        reader_rt[cx][cy] = [w.buffer_address(), *cols[j]]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "wq_b_linear_reader.cpp"),
        core_ranges=worker_crs,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        defines=_ablate_defines("w_read"),
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Worker writer (x round arrival + output store), BRISC / NoC1 ----
    writer_ct = [CB_X_RESIDENT, CB_OUT_BLOCK, block_m_tiles, block_k_tiles, block_n_tiles, num_k_blocks, tensor_n_tiles]
    writer_ct.append(int(Y_STORE_NOC_SPLIT))
    writer_ct.extend(ttnn.TensorAccessorArgs(y).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    for j, (cx, cy) in enumerate(workers):
        writer_rt[cx][cy] = [y.buffer_address(), *cols[j]]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "wq_b_linear_writer.cpp"),
        core_ranges=worker_crs,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        defines=_ablate_defines("y_store"),
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Worker compute (one matmul_block over num_k_blocks K-steps) ----
    compute_ct = [
        CB_X_RESIDENT,
        CB_W_BLOCK,
        CB_OUT_BLOCK,
        in0_num_subblocks,
        in1_num_subblocks,
        sb_h,
        sb_w,
        block_k_tiles,
        num_k_blocks,
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "wq_b_linear_compute.cpp"),
        core_ranges=worker_crs,
        compile_time_args=compute_ct,
        runtime_args=[],
        defines=_ablate_defines("compute"),
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=compute_config.math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_approx_mode=compute_config.math_approx_mode,
        ),
    )

    # ---- x multicast channels: one per sender, rect receivers, sender outside the rect, no handshake.
    # Channel s is attached as X_MCAST_PREFIXES[s] to sender s's kernel and to the worker writer; the
    # other prefixes are attached absent (each sender kernel uses its one active slot).
    descriptor = ttnn.ProgramDescriptor(cbs=[cb_x_resident, cb_w_block, cb_out_block])
    next_sem = 0
    for s_idx, (scx, scy) in enumerate(senders):
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=X_SENDER_NOCS[s_idx], handshake=False, data_ready=ttnn.McastDataReady.Counter),
            rect,
            num_workers,
            ttnn.McastExplicitFixedSenderConfig([ttnn.CoreCoord(scx, scy)]),
        )
        targets = [sender_kernels[s_idx], writer_kernel] + ([reader_kernel] if s_idx == 0 else [])
        mcast.attach(descriptor, X_MCAST_PREFIXES[s_idx], targets, next_sem)
        assert next_sem == s_idx and mcast.next_semaphore_id() == s_idx + 1, "x channel semaphore ids drifted"
        next_sem = mcast.next_semaphore_id()
    for s_idx, prefix in enumerate(X_MCAST_PREFIXES):
        for k_idx, kernel in enumerate(sender_kernels):
            if k_idx != s_idx:
                ttnn.attach_absent_mcast(kernel, prefix)
        if s_idx >= len(senders):
            ttnn.attach_absent_mcast(writer_kernel, prefix)
    extra = []
    if helper is not None:
        # The cell semaphore id is the next free one; patch it into the sender reader's define.
        sem_id = next_sem
        for kd in x_reader_kernels:
            kd.defines = [(n, str(sem_id) if v == "XHSEM" else v) for n, v in kd.defines]
        sems = list(descriptor.semaphores)
        sems.append(ttnn.SemaphoreDescriptor(id=sem_id, core_ranges=_crs([senders[0], helper]), initial_value=0))
        descriptor.semaphores = sems
        hx, hy = helper
        snoc = device.worker_core_from_logical_core(ttnn.CoreCoord(*senders[0]))
        h_ct = [CB_X_RESIDENT, block_m_tiles, block_k_tiles, num_k_blocks, tensor_k_tiles, x_read_trids, 1, 2, sem_id]
        h_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        h_rt = ttnn.RuntimeArgs()
        h_rt[hx][hy] = [x.buffer_address(), snoc.x, snoc.y]
        extra.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "wq_b_linear_x_helper.cpp"),
                core_ranges=_crs([helper]),
                compile_time_args=h_ct,
                runtime_args=h_rt,
                defines=_ablate_defines("x_read"),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                ),
            )
        )
    descriptor.kernels = [*x_reader_kernels, *sender_kernels, reader_kernel, writer_kernel, compute_kernel, *extra]
    return descriptor
