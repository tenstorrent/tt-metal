# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""wq_b_linear program descriptor — regime R1 `n_split_x_bcast_resident`.

n is split across the full device grid; x is read once from DRAM by per-round injectors and
multicast into every core's resident cb_x_resident; each core streams its own W columns and
accumulates its (m, core_n) output block in place with packer L1 accumulation.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

K_BLOCK_TILES = 3
MIN_CORES = 16
MAX_READ_TRIDS = 15
W_READS_IN_FLIGHT = 1
# Compute K-blocks per x multicast round (raised if needed to fit MAX_X_MCAST_CHANNELS); one injector per round.
X_ROUND_K_BLOCKS = 4
# "per_round": one fixed-sender Flag channel per round, each fired once per launch, so no ready handshake.
# "rotating": one channel with rotating senders; Flag + ready handshake when chained, Counter when concurrent.
X_MCAST_CHANNELS = "per_round"
# False: injector r multicasts once it has received round r - 1 (one multicast on the NoC at a time).
# True: every injector multicasts as soon as its prefetch lands.
X_ROUNDS_CONCURRENT = False
# "diagonal": injectors on distinct columns of the last rows (away from the row-major-first heavy n cores),
#   so their x prefetches share no row links.
# "last_row": the last num_x_injectors cores in row-major order.
X_INJECTOR_PLACEMENT = "diagonal"
# Must match the channels declared in the writer kernel.
MAX_X_MCAST_CHANNELS = 8

CB_X_RESIDENT = 0
CB_W_STREAM = 1
CB_Y_TILES = 16

MCAST_FIRST_SEMAPHORE_ID = 0


def _largest_divisor_at_most(n, cap):
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _dest_limit(cfg):
    limit = 16 if cfg.dst_full_sync_en else 8
    return limit // 2 if cfg.fp32_dest_acc_en else limit


def _subblock(block_m_tiles, core_n_tiles, dest_limit):
    best = (1, 1)
    for sb_h in range(1, block_m_tiles + 1):
        if block_m_tiles % sb_h:
            continue
        for sb_w in range(1, core_n_tiles + 1):
            if core_n_tiles % sb_w or sb_h * sb_w > dest_limit:
                continue
            area, best_area = sb_h * sb_w, best[0] * best[1]
            if area > best_area or (area == best_area and sb_h > best[0]):
                best = (sb_h, sb_w)
    return best


def _x_round_k_blocks(num_k_blocks):
    d = _largest_divisor_at_most(num_k_blocks, X_ROUND_K_BLOCKS)
    while num_k_blocks // d > MAX_X_MCAST_CHANNELS:
        d = next(c for c in range(d + 1, num_k_blocks + 1) if num_k_blocks % c == 0)
    return d


def _x_injectors(grid, num_injectors):
    assert X_INJECTOR_PLACEMENT in ("diagonal", "last_row"), X_INJECTOR_PLACEMENT
    if X_INJECTOR_PLACEMENT == "diagonal" and num_injectors <= min(grid.x, grid.y):
        first_row = grid.y - num_injectors
        return [ttnn.CoreCoord(r * grid.x // num_injectors, first_row + r) for r in range(num_injectors)]
    grid_row_major = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    return grid_row_major[len(grid_row_major) - num_injectors :]


def _attach_x_mcast(descriptor, device, writer_kernel, grid_cores, injectors, num_x_rounds):
    """Attach channels x_mcast_0 .. x_mcast_{MAX_X_MCAST_CHANNELS-1}; unused ones are absent."""
    num_grid_cores = grid_cores.num_cores()
    if X_MCAST_CHANNELS == "rotating":
        if X_ROUNDS_CONCURRENT:
            cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=False, data_ready=ttnn.McastDataReady.Counter)
        else:
            cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True, data_ready=ttnn.McastDataReady.Flag)
        injector_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in injectors])
        sender = ttnn.McastRotatingSenderConfig(injector_cores, sender_order=ttnn.McastCoreOrder.RowMajor)
        ttnn.Mcast(device, cfg, grid_cores, num_grid_cores, sender).attach(
            descriptor, "x_mcast_0", [writer_kernel], MCAST_FIRST_SEMAPHORE_ID
        )
        num_present = 1
    else:
        assert X_MCAST_CHANNELS == "per_round", X_MCAST_CHANNELS
        cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=False, data_ready=ttnn.McastDataReady.Flag)
        grid_row_major = ttnn.corerange_to_cores(grid_cores, row_wise=True)
        index_of = {(c.x, c.y): i for i, c in enumerate(grid_row_major)}
        semaphore_id = MCAST_FIRST_SEMAPHORE_ID
        for r in range(num_x_rounds):
            sender = ttnn.McastFixedSenderConfig(sender_index=index_of[(injectors[r].x, injectors[r].y)])
            mcast = ttnn.Mcast(device, cfg, grid_cores, num_grid_cores, sender)
            mcast.attach(descriptor, f"x_mcast_{r}", [writer_kernel], semaphore_id)
            semaphore_id = mcast.next_semaphore_id()
        num_present = num_x_rounds
    for r in range(num_present, MAX_X_MCAST_CHANNELS):
        ttnn.attach_absent_mcast(writer_kernel, f"x_mcast_{r}")


def _copy_compute_config(cfg):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=cfg.math_fidelity,
        fp32_dest_acc_en=cfg.fp32_dest_acc_en,
        dst_full_sync_en=cfg.dst_full_sync_en,
        math_approx_mode=cfg.math_approx_mode,
    )


def create_program_descriptor(x_tensor, w_tensor, y_tensor, compute_cfg):
    device = x_tensor.device()
    tile_h, tile_w = x_tensor.tile.tile_shape

    tensor_m_tiles = (x_tensor.shape[0] * x_tensor.shape[1] * x_tensor.shape[2] + tile_h - 1) // tile_h
    tensor_k_tiles = (x_tensor.shape[-1] + tile_w - 1) // tile_w
    tensor_n_tiles = (w_tensor.shape[-1] + tile_w - 1) // tile_w

    block_m_tiles = tensor_m_tiles
    k_block_tiles = _largest_divisor_at_most(tensor_k_tiles, K_BLOCK_TILES)
    num_k_blocks = tensor_k_tiles // k_block_tiles
    w_reads_in_flight = min(num_k_blocks, W_READS_IN_FLIGHT, MAX_READ_TRIDS)
    x_round_k_blocks = _x_round_k_blocks(num_k_blocks)
    num_x_rounds = num_k_blocks // x_round_k_blocks

    grid = device.compute_with_storage_grid_size()
    grid_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    num_grid_cores = grid.x * grid.y

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        n_tiles_g1,
        n_tiles_g2,
    ) = ttnn.split_work_to_cores(grid, tensor_n_tiles, row_wise=True)
    core_n_tiles_max = max(n_tiles_g1, n_tiles_g2)

    grid_row_major = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    if X_MCAST_CHANNELS == "per_round":
        num_x_injectors = num_x_rounds
    else:
        num_x_injectors = max(2, min(num_x_rounds, grid.x))
    x_injectors = _x_injectors(grid, num_x_injectors)

    x_page_bytes = x_tensor.buffer_page_size()
    w_page_bytes = w_tensor.buffer_page_size()
    y_page_bytes = y_tensor.buffer_page_size()

    # x and W are whole-K resident: the writer and reader reserve all num_k_blocks blocks up front.
    x_cb_pages = block_m_tiles * k_block_tiles * num_k_blocks
    w_cb_pages = k_block_tiles * num_k_blocks * core_n_tiles_max
    y_cb_pages = block_m_tiles * core_n_tiles_max

    def cb(index, pages, page_bytes, dtype, cores):
        return ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
        )

    cbs = [
        cb(CB_X_RESIDENT, x_cb_pages, x_page_bytes, x_tensor.dtype, grid_cores),
        cb(CB_W_STREAM, w_cb_pages, w_page_bytes, w_tensor.dtype, all_cores),
        cb(CB_Y_TILES, y_cb_pages, y_page_bytes, y_tensor.dtype, all_cores),
    ]

    # ---- per-core n assignment (row-major prefix sum) ----
    assignment = {}
    n_start = 0
    for group, n_tiles in ((core_group_1, n_tiles_g1), (core_group_2, n_tiles_g2)):
        if n_tiles == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            assignment[(core.x, core.y)] = (n_start, n_tiles)
            n_start += n_tiles
    assert n_start == tensor_n_tiles

    # ---- reader: W stream (NCRISC, NoC0) ----
    reader_ct = [CB_W_STREAM, tensor_n_tiles, k_block_tiles, num_k_blocks, w_reads_in_flight]
    reader_ct.extend(ttnn.TensorAccessorArgs(w_tensor).get_compile_time_args())
    reader_rt = ttnn.RuntimeArgs()
    for (cx, cy), (core_n_start, core_n) in assignment.items():
        reader_rt[cx][cy] = [w_tensor.buffer_address(), core_n_start, core_n]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "wq_b_linear_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
    )

    # ---- writer: x broadcast + y store (BRISC, NoC1), full grid ----
    writer_ct = [
        CB_X_RESIDENT,
        CB_Y_TILES,
        block_m_tiles,
        tensor_k_tiles,
        k_block_tiles,
        x_round_k_blocks,
        num_x_rounds,
        tensor_n_tiles,
        int(X_MCAST_CHANNELS == "per_round"),
        int(X_ROUNDS_CONCURRENT),
        MAX_X_MCAST_CHANNELS,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(x_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(y_tensor).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    for core in grid_row_major:
        core_n_start, core_n = assignment.get((core.x, core.y), (0, 0))
        writer_rt[core.x][core.y] = [x_tensor.buffer_address(), y_tensor.buffer_address(), core_n_start, core_n]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "wq_b_linear_writer.cpp"),
        core_ranges=grid_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    )

    # ---- compute: one kernel per core group (core_n_tiles and subblocks are CT) ----
    dest_limit = _dest_limit(compute_cfg)
    compute_kernels = []
    for group, n_tiles in ((core_group_1, n_tiles_g1), (core_group_2, n_tiles_g2)):
        if n_tiles == 0 or group.num_cores() == 0:
            continue
        sb_h, sb_w = _subblock(block_m_tiles, n_tiles, dest_limit)
        compute_ct = [
            CB_X_RESIDENT,
            CB_W_STREAM,
            CB_Y_TILES,
            block_m_tiles,
            n_tiles,
            sb_h,
            sb_w,
            k_block_tiles,
            num_k_blocks,
        ]
        compute_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "wq_b_linear_compute.cpp"),
                core_ranges=group,
                compile_time_args=compute_ct,
                runtime_args=[],
                config=_copy_compute_config(compute_cfg),
            )
        )

    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)

    _attach_x_mcast(descriptor, device, writer_kernel, grid_cores, x_injectors, num_x_rounds)

    descriptor.kernels = [reader_kernel, writer_kernel, *compute_kernels]
    return descriptor
