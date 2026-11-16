# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""wq_b_linear program descriptor — regime R1 `n_split_x_bcast_resident` (R4 `n_split_k_remainder_combine` behind
K_REMAINDER_SPLIT).

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
# Read depth on the cores holding the n remainder (the larger n share); 0 = W_READS_IN_FLIGHT.
W_READS_IN_FLIGHT_G1 = 0
# n tiles per active core; 0 = fill the grid. Fewer, evenly loaded cores trade grid width for no light/heavy tail.
CORE_N_TILES = 2
# "row_major": active cores and the n remainder take the row-major-first cores; "spread": both evenly strided.
CORE_PLACEMENT = "row_major"
# Give each run of num_dram_banks equally loaded multi-column cores bank-strided columns (n, n + banks, ...):
# a core's W tiles of one k row are then adjacent in one bank and arrive in one transfer.
W_BANK_COALESCE = True
# Last K-blocks of each core's W read by the writer (BRISC) on NoC1, beside the reader's NoC0 stream.
W_NOC1_K_BLOCKS = 0
# True: the writer issues those reads after its x phase instead of beside the x prefetch.
W_NOC1_AFTER_X = True
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
# Regime R4: every core owns one full column; the columns left over after one per core are cut into K-slices
# spread over the cores. Slice cores unicast their partial to the column owner (the slice-0 core), which sums
# its own partial and the landed ones in slice order. Falls back to the CORE_N_TILES split when the grid is too
# small for two slices per leftover column. Parked off: 22.7 us vs 22.4 us for the CORE_N_TILES split, because
# 110 single-column cores stream W (1088 B reads) slower than 65 bank-coalesced two-column cores (2176 B reads).
K_REMAINDER_SPLIT = False
# A core computes its K-slice after main K-block min(slice end, num_k_blocks - 1 - SLICE_EARLY_K_BLOCKS), so its
# partial reaches the owner before the owner's combine.
SLICE_EARLY_K_BLOCKS = 9
# The owner combines after main K-block num_k_blocks - 1 - COMBINE_EARLY_K_BLOCKS (not before its own slice),
# inside the W-bound main loop instead of on the tail after the last W block.
COMBINE_EARLY_K_BLOCKS = 6
# Tenths of each core's y block the reader (NCRISC, NoC0; idle once its W stream ends) stores beside the writer's
# NoC1 share; 0 = the writer stores it all. Leftover-column tiles (R4) stay on the writer.
Y_READER_SHARE_TENTHS = 4
# Last K-block y release: "none" = one push after the whole block; "column" = one push per column subblock group;
# "row" = one push per tile row (last K-block computed as 1 x row_sb_w subblocks).
Y_RELEASE = "column"
_Y_RELEASE_MODE = {"none": 0, "column": 1, "row": 2}
# True: the reader stores the earliest-released whole groups instead of the tail of every group.
Y_SPLIT_FRONT = False

CB_X_RESIDENT = 0
CB_W_STREAM = 1
CB_X_SLICE = 2
CB_W_SLICE = 3
CB_Y_TILES = 16
CB_PARTIAL = 17
CB_LANDING = 18

W_HANDOFF_SEMAPHORE_ID = 0
LANDING_SEMAPHORE_ID = 1
Y_DONE_SEMAPHORE_ID = 2
MCAST_FIRST_SEMAPHORE_ID = 3


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


def _strided_pick(items, count):
    return [items[i * len(items) // count] for i in range(count)]


def _core_n_counts(num_grid_cores, tensor_n_tiles, num_banks):
    """Per-core n shares, largest first."""
    if CORE_N_TILES > 1 and W_BANK_COALESCE:
        runs, rem = divmod(tensor_n_tiles, num_banks * CORE_N_TILES)
        # Columns left over after the bank runs cannot pair within one bank; they go one per core.
        if runs * num_banks + rem <= num_grid_cores:
            return [CORE_N_TILES] * (runs * num_banks) + [1] * rem
    num_cores = min(num_grid_cores, tensor_n_tiles)
    if CORE_N_TILES:
        num_cores = min(num_cores, -(-tensor_n_tiles // CORE_N_TILES))
    base, rem = divmod(tensor_n_tiles, num_cores)
    return [base + 1] * rem + [base] * (num_cores - rem)


def _split_n(grid, tensor_n_tiles, num_banks):
    """Returns [(cores, n_tiles)] per group, the larger n share first (split_work_to_cores convention)."""
    assert CORE_PLACEMENT in ("row_major", "spread"), CORE_PLACEMENT
    grid_row_major = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    counts = _core_n_counts(len(grid_row_major), tensor_n_tiles, num_banks)
    num_cores = len(counts)
    num_heavy = counts.count(counts[0]) if counts[-1] != counts[0] else 0
    if CORE_PLACEMENT == "spread":
        active = _strided_pick(grid_row_major, num_cores)
        heavy_idx = {i * num_cores // num_heavy for i in range(num_heavy)} if num_heavy else set()
    else:
        active = grid_row_major[:num_cores]
        heavy_idx = set(range(num_heavy))
    heavy = [c for i, c in enumerate(active) if i in heavy_idx]
    light = [c for i, c in enumerate(active) if i not in heavy_idx]
    return [(heavy, counts[0]), (light, counts[-1])] if num_heavy else [(light, counts[0])]


def _assign_columns(n_groups, tensor_n_tiles, num_banks):
    """{(x, y): (n_start, n_tiles, n_stride)}; the core owns columns n_start + j * n_stride."""
    ordered = [(c, n_tiles) for cores, n_tiles in n_groups for c in cores]
    assignment = {}
    n_start = i = 0
    while i < len(ordered):
        n_tiles = ordered[i][1]
        run = ordered[i : i + num_banks]
        if W_BANK_COALESCE and n_tiles > 1 and len(run) == num_banks and all(n == n_tiles for _, n in run):
            for j, (c, _) in enumerate(run):
                assignment[(c.x, c.y)] = (n_start + j, n_tiles, num_banks)
            n_start += num_banks * n_tiles
            i += num_banks
        else:
            c = ordered[i][0]
            assignment[(c.x, c.y)] = (n_start, n_tiles, 1)
            n_start += n_tiles
            i += 1
    assert n_start == tensor_n_tiles
    return assignment


def _k_remainder_split(grid, tensor_n_tiles, num_k_blocks):
    """R4 split, or None when it does not apply.

    Returns (cores, slices): core i owns column i; slices[i] = (column, kb_start, kb_count, s, num_slices, owner)
    or None. Slice s of a leftover column covers K-blocks [s * nkb // S, (s + 1) * nkb // S); its owner is the
    core holding slice 0.
    """
    grid_row_major = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    num_cores = min(len(grid_row_major), tensor_n_tiles)
    num_leftover = tensor_n_tiles - num_cores
    num_slices = min(num_k_blocks, num_cores // num_leftover) if num_leftover else 0
    if not K_REMAINDER_SPLIT or (num_leftover and num_slices < 2):
        return None
    cores = grid_row_major[:num_cores]
    slices = [None] * num_cores
    for j in range(num_leftover):
        owner = cores[j * num_slices]
        for s in range(num_slices):
            kb_start = s * num_k_blocks // num_slices
            kb_end = (s + 1) * num_k_blocks // num_slices
            slices[j * num_slices + s] = (num_cores + j, kb_start, kb_end - kb_start, s, num_slices, owner)
    return cores, slices


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


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
    w_noc1_k_blocks = min(W_NOC1_K_BLOCKS, num_k_blocks - 1)
    x_round_k_blocks = _x_round_k_blocks(num_k_blocks)
    num_x_rounds = num_k_blocks // x_round_k_blocks

    grid = device.compute_with_storage_grid_size()
    grid_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    num_grid_cores = grid.x * grid.y

    num_dram_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    k_split = _k_remainder_split(grid, tensor_n_tiles, num_k_blocks)
    if k_split:
        k_cores, slices = k_split
        n_groups = [(k_cores, 1)]
        assignment = {(c.x, c.y): (i, 1, 1) for i, c in enumerate(k_cores)}
        slice_of = {(c.x, c.y): sl for c, sl in zip(k_cores, slices)}
    else:
        n_groups = _split_n(grid, tensor_n_tiles, num_dram_banks)
        assignment = _assign_columns(n_groups, tensor_n_tiles, num_dram_banks)
        slice_of = {}
    has_slices = any(slice_of.values())
    all_cores = _core_range_set([c for cores, _ in n_groups for c in cores])
    core_n_tiles_max = n_groups[0][1]
    max_slice_k_blocks = max((sl[2] for sl in slice_of.values() if sl), default=0)
    num_landing = max((sl[4] - 1 for sl in slice_of.values() if sl), default=0)

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
    # Owners hold their own column plus the combined leftover column.
    y_cb_pages = block_m_tiles * (core_n_tiles_max + int(has_slices))

    def cb(indices, pages, page_bytes, dtype, cores):
        return ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page_bytes) for i in indices
            ],
        )

    # cb_x_slice aliases cb_x_resident: the K-slice matmul reads x through its own FIFO pointers.
    cbs = [
        cb(
            [CB_X_RESIDENT, CB_X_SLICE] if has_slices else [CB_X_RESIDENT],
            x_cb_pages,
            x_page_bytes,
            x_tensor.dtype,
            grid_cores,
        ),
        cb([CB_W_STREAM], w_cb_pages, w_page_bytes, w_tensor.dtype, all_cores),
        cb([CB_Y_TILES], y_cb_pages, y_page_bytes, y_tensor.dtype, all_cores),
    ]
    if has_slices:
        cbs += [
            cb([CB_W_SLICE], k_block_tiles * max_slice_k_blocks, w_page_bytes, w_tensor.dtype, all_cores),
            cb([CB_PARTIAL], block_m_tiles, y_page_bytes, y_tensor.dtype, all_cores),
            # On the full grid so the landing address is the same on every core (senders use their own).
            cb([CB_LANDING], block_m_tiles * num_landing, y_page_bytes, y_tensor.dtype, grid_cores),
        ]

    def slice_args(core):
        """(column, kb_start, kb_count, s, num_slices, owner noc x, owner noc y); the owner holds slice 0."""
        sl = slice_of.get((core.x, core.y))
        if not sl:
            return [0, 0, 0, 0, 0, 0, 0]
        column, kb_start, kb_count, s, num_slices, owner = sl
        owner_noc = device.worker_core_from_logical_core(owner)
        return [column, kb_start, kb_count, s, num_slices, owner_noc.x, owner_noc.y]

    def slice_after_block(core):
        sl = slice_of.get((core.x, core.y))
        if not sl:
            return 0
        return max(0, min(sl[1] + sl[2] - 1, num_k_blocks - 1 - SLICE_EARLY_K_BLOCKS))

    # ---- reader: W stream (NCRISC, NoC0), one kernel per core group (read depth is CT) ----
    reader_kernels = []
    for gi, (cores, _) in enumerate(n_groups):
        depth = W_READS_IN_FLIGHT_G1 if gi == 0 and len(n_groups) > 1 and W_READS_IN_FLIGHT_G1 else W_READS_IN_FLIGHT
        reader_ct = [
            CB_W_STREAM,
            tensor_n_tiles,
            k_block_tiles,
            num_k_blocks,
            min(num_k_blocks - w_noc1_k_blocks, depth, MAX_READ_TRIDS),
            w_noc1_k_blocks,
            W_HANDOFF_SEMAPHORE_ID,
            CB_W_SLICE,
            CB_Y_TILES,
            block_m_tiles,
            Y_READER_SHARE_TENTHS,
            Y_DONE_SEMAPHORE_ID,
            _Y_RELEASE_MODE[Y_RELEASE],
            int(Y_SPLIT_FRONT),
        ]
        reader_ct.extend(ttnn.TensorAccessorArgs(w_tensor).get_compile_time_args())
        reader_ct.extend(ttnn.TensorAccessorArgs(y_tensor).get_compile_time_args())
        reader_rt = ttnn.RuntimeArgs()
        for c in cores:
            reader_rt[c.x][c.y] = [
                w_tensor.buffer_address(),
                *assignment[(c.x, c.y)],
                *slice_args(c)[:3],
                y_tensor.buffer_address(),
            ]
        reader_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "wq_b_linear_reader.cpp"),
                core_ranges=_core_range_set(cores),
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                ),
            )
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
        CB_W_STREAM,
        w_noc1_k_blocks,
        W_HANDOFF_SEMAPHORE_ID,
        int(W_NOC1_AFTER_X),
        int(has_slices),
        CB_X_SLICE,
        CB_PARTIAL,
        CB_LANDING,
        LANDING_SEMAPHORE_ID,
        Y_READER_SHARE_TENTHS,
        Y_DONE_SEMAPHORE_ID,
        _Y_RELEASE_MODE[Y_RELEASE],
        int(Y_SPLIT_FRONT),
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(x_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(y_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(w_tensor).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    for core in grid_row_major:
        writer_rt[core.x][core.y] = [
            x_tensor.buffer_address(),
            y_tensor.buffer_address(),
            *assignment.get((core.x, core.y), (0, 0, 1)),
            w_tensor.buffer_address(),
            *slice_args(core),
        ]
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
    for cores, n_tiles in n_groups:
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
            int(has_slices),
            CB_X_SLICE,
            CB_W_SLICE,
            CB_PARTIAL,
            CB_LANDING,
            _Y_RELEASE_MODE[Y_RELEASE],
            _largest_divisor_at_most(n_tiles, dest_limit),
        ]
        compute_rt = ttnn.RuntimeArgs()
        for c in cores:
            _, kb_start, kb_count, s, num_slices, _, _ = slice_args(c)
            num_landed = num_slices - 1 if kb_count and s == 0 else 0
            combine_after = max(slice_after_block(c), num_k_blocks - 1 - COMBINE_EARLY_K_BLOCKS)
            compute_rt[c.x][c.y] = [kb_start, kb_count, slice_after_block(c), num_landed, combine_after]
        compute_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "wq_b_linear_compute.cpp"),
                core_ranges=_core_range_set(cores),
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=_copy_compute_config(compute_cfg),
            )
        )

    semaphores = [
        ttnn.SemaphoreDescriptor(id=W_HANDOFF_SEMAPHORE_ID, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=LANDING_SEMAPHORE_ID, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=Y_DONE_SEMAPHORE_ID, core_ranges=all_cores, initial_value=0),
    ]
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)

    _attach_x_mcast(descriptor, device, writer_kernel, grid_cores, x_injectors, num_x_rounds)

    descriptor.kernels = [*reader_kernels, writer_kernel, *compute_kernels]
    return descriptor
