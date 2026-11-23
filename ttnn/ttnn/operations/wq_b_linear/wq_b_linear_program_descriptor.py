# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for wq_b_linear (N-split, x multicast resident, W streamed per worker).

Cores:
  * 1 x-sender (outside the receiver rectangle): reads x from DRAM one round
    (block_m_tiles x block_k_tiles tiles) at a time into its cb_x_resident slot r and
    multicasts the slot to the same address on every worker (Counter data-ready, no handshake).
  * num_workers workers (one rectangle): reader streams the worker's W columns per K-step,
    writer waits for each x round and pushes it to compute, compute runs one matmul_block over
    all K-steps, writer stores the 5 x block_n_tiles output block into the L1-interleaved y.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32

# ---- Blocking knobs (single source of truth; see op_design.md "Blocking Model") ----
BLOCK_N_TILES = 2  # output columns per worker; num_workers = tensor_n_tiles / BLOCK_N_TILES
BLOCK_K_TILES = 4  # K tiles per K-step (matmul in0_block_k); num_k_blocks derived
W_DEPTH_BLOCKS = None  # cb_w_block depth in K-steps; None = num_k_blocks (whole W slice resident)
OUT_DEPTH_BLOCKS = 1  # one output block per core
MCAST_RECT_WIDTH_MAX = 8
DEST_TILE_LIMIT = 8  # 16-bit DEST, half-sync (DEST_AUTO_LIMIT)

# ---- CB indices (semantic names) ----
CB_X_RESIDENT = 0
CB_W_BLOCK = 1
CB_INTERM_ACC = 2
CB_OUT_BLOCK = 3


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
    rect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + rect_w - 1, grid.y - 1))])
    sender = (grid.x - 1, grid.y - 1)
    return workers, rect, sender


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

    sb_h, sb_w = _choose_subblock(block_m_tiles, block_n_tiles, DEST_TILE_LIMIT)
    in0_num_subblocks = block_m_tiles // sb_h
    in1_num_subblocks = block_n_tiles // sb_w

    workers, rect, sender = _worker_geometry(device, num_workers)
    worker_crs = _crs(workers)
    sender_crs = _crs([sender])
    all_crs = _crs(workers + [sender])

    num_banks = device.dram_grid_size().x
    cols = _bank_spread_columns(tensor_n_tiles, block_n_tiles, num_workers, num_banks)

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
    cb_interm_acc = ttnn.CBDescriptor(
        total_size=out_block_tiles * y_tile_bytes,
        core_ranges=worker_crs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_INTERM_ACC, data_format=y.dtype, page_size=y_tile_bytes)
        ],
    )
    cb_out_block = ttnn.CBDescriptor(
        total_size=out_block_tiles * OUT_DEPTH_BLOCKS * y_tile_bytes,
        core_ranges=worker_crs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUT_BLOCK, data_format=y.dtype, page_size=y_tile_bytes)
        ],
    )

    # ---- Sender kernel (x DRAM -> slot -> multicast), BRISC / NoC1 ----
    sender_ct = [CB_X_RESIDENT, block_m_tiles, block_k_tiles, num_k_blocks, tensor_k_tiles, x_tile_bytes]
    sender_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[sender[0]][sender[1]] = [x.buffer_address()]
    sender_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "wq_b_linear_sender.cpp"),
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        runtime_args=sender_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    )

    # ---- Worker reader (W columns per K-step), NCRISC / NoC0 ----
    reader_ct = [CB_W_BLOCK, block_k_tiles, block_n_tiles, num_k_blocks, tensor_n_tiles]
    reader_ct.extend(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    reader_rt = ttnn.RuntimeArgs()
    for j, (cx, cy) in enumerate(workers):
        reader_rt[cx][cy] = [w.buffer_address(), *cols[j]]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "wq_b_linear_reader.cpp"),
        core_ranges=worker_crs,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Worker writer (x round arrival + output store), BRISC / NoC1 ----
    writer_ct = [CB_X_RESIDENT, CB_OUT_BLOCK, block_m_tiles, block_k_tiles, block_n_tiles, num_k_blocks, tensor_n_tiles]
    writer_ct.extend(ttnn.TensorAccessorArgs(y).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    for j, (cx, cy) in enumerate(workers):
        writer_rt[cx][cy] = [y.buffer_address(), *cols[j]]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "wq_b_linear_writer.cpp"),
        core_ranges=worker_crs,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Worker compute (one matmul_block over num_k_blocks K-steps) ----
    compute_ct = [
        CB_X_RESIDENT,
        CB_W_BLOCK,
        CB_INTERM_ACC,
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
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=compute_config.math_fidelity,
            fp32_dest_acc_en=False,
            math_approx_mode=compute_config.math_approx_mode,
        ),
    )

    # ---- x multicast channel: rect receivers, explicit sender outside the rect, no handshake ----
    mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=False, data_ready=ttnn.McastDataReady.Counter),
        rect,
        num_workers,
        ttnn.McastExplicitFixedSenderConfig([ttnn.CoreCoord(*sender)]),
    )

    descriptor = ttnn.ProgramDescriptor(cbs=[cb_x_resident, cb_w_block, cb_interm_acc, cb_out_block])
    mcast.attach(descriptor, "x_mcast", [sender_kernel, writer_kernel], 0)
    descriptor.kernels = [sender_kernel, reader_kernel, writer_kernel, compute_kernel]
    return descriptor
