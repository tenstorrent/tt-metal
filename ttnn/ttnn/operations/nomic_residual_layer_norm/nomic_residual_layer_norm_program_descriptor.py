# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for nomic_residual_layer_norm (op_design.md § Circular Buffers / § Work Distribution).

Every CB size, loop count and the grid derive from the Plan (`_plan()` in the op module): block_rows,
hidden_tiles, stat_tiles, hidden_split, num_row_blocks. Nothing here is sized from a whole-op dimension.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic names; op_design.md § Circular Buffers).
CB_X = 0
CB_RESIDUAL = 1
CB_GAMMA = 2
CB_BETA = 3
CB_Y = 5
CB_ROW_PARTIALS = 6  # aliases CB_RSTD_COL (disjoint lifetimes)
CB_STATS_LOCAL = 8
CB_STATS_GATHER = 9
CB_CENTER_COL = 10  # per-row mean, col 0 (SEEDED_NORMALIZE: -mean * rstd, the stage-1 seed)
CB_RSTD_COL = 11
CB_OUTPUT = 12

FP32_TILE = 4096
BF16_TILE = 2048


# Every CB whose only reader is copy_tile / fp32 unary_bcast (statistics never pass through srcA/srcB). The seeded
# normalize reads cb_center_col only as its unary-broadcast seed (exact fp32 unpack-to-dest); the default normalize
# reads it as an FPU srcB, so it stays off the list then.
def _unpack_to_dest_fp32_cbs():
    return (CB_ROW_PARTIALS, CB_STATS_GATHER) + ((CB_CENTER_COL,) if SEEDED_NORMALIZE else ())


NUM_CIRCULAR_BUFFERS = 64

MCAST_PREFIX = "stats"
MCAST_FIRST_SEMAPHORE_ID = 0
STATS_ARRIVED_SEMAPHORE_ID = 0  # concurrent_unicast only (no Mcast channel is attached then)
EXCHANGE_MODES = ("serialized_chain", "serialized_mcast", "concurrent_mcast", "concurrent_unicast")
WRITER_NOC = ttnn.NOC.NOC_0

# Tiles per DEST window in the streaming element-wise stages (residual add, 4 normalize stages). fp32 DEST in
# half-sync holds 4 tiles; 1 is the per-tile walk.
ELTWISE_BLOCK_TILES = 4

# Exchange transport knob (perf lamps L5 / L4). All three are correct; the default is the fastest measured.
#   "serialized_chain": handshaked fixed-sender channels, member-ordered rounds, ChainUnicast for row-wrapping
#                       groups (2026-11-03, M=128: exchange ~2.3 us).
#   "serialized_mcast": the same rounds over hardware multicast rectangles (measured ~3.1 us avg / 6.3 us max).
#   "concurrent_mcast": no handshake, every member multicasts at once (one round per call only; measured slower:
#                       ~3.8 us avg / 10 us max, 96 concurrent multicasts contend on the NoC).
#   "concurrent_unicast": op_design lamp L5 -- every member unicasts its slot to its peers and bumps one arrival
#                       semaphore on each, all at once (one round per call only). Raw NoC writes + one semaphore:
#                       the Mcast helper has no handshake-free unicast form (chain needs the handshake).
# Refinement 3 re-measured the op layout (device us, residual): with one round per call concurrent_unicast wins
# (2x512 7.30 -> 7.00, 4x512 10.42 -> 10.00), concurrent_mcast loses (10.5 / 16.9). A plan with several rounds (every
# 4-row shard since the pipelined row blocks) falls back to serialized_chain below, whose rounds the lookahead hides.
EXCHANGE_MODE = "concurrent_unicast"
# The same knob for the model layout's groups (one grid column of 8 cores: a 1 x 8 rectangle, so hardware multicast
# is legal there too). See the Refinement 1 changelog entry for the measurement behind the default.
# Re-measured in Refinement 3 on single-round model plans: concurrent_unicast 4x288 15.2 -> 17.0, 2x512 13.3 -> 15.0
# (8 members each unicasting to 7 peers at once contend); concurrent_mcast 18.7-20.4 / 17.0-19.1.
MODEL_EXCHANGE_MODE = "serialized_chain"


# Row-block software pipelining (perf lamp L2): with more than one row block, phase A of block b+1 runs before block
# b's finalize so the writer's exchange of block b overlaps it (normalize recomputes y = x + r from the resident
# shards, so cb_y stays one block). Costs cb_stats_local a second block.
# No effect on a one-block plan (every op_layout plan at the default knobs).
LOOKAHEAD = True

# Normalize as two seeded multiplies (Refinement 2): z = (-mean * rstd) + y * rstd, out = beta + z * gamma, with the
# seed unary-broadcast into the DEST lane the ELWMUL then accumulates onto -- 2 fp32 L1 round trips a tile instead of
# 4. A measured regression, so off: the -mean * rstd seed must land in DEST in full fp32 (unpack-to-dest, so
# cb_center_col joins _unpack_to_dest_fp32_cbs() above), because rows with |mean| >> std cancel y * rstd against it
# (constant rows: an srcB-truncated seed is off by up to 2.9). That fp32 column-broadcast unpack-to-dest is slow
# (op 8x512 residual 23.3 vs 18.1 us). Seeding through srcB instead is a null even before it fails the constant-row
# test (18.28 us): the srcB unpack + MOVB2D of a seed costs about what the add stage it replaces costs.
SEEDED_NORMALIZE = False


# Lookahead plans with a residual recompute y = x + r in normalize (cb_y holds one block). Fused: normalize stage 1 is
# one chain (x - mean) + r (BinaryFpu Sub + DestReuseBinary Add, the library layernorm's pre-add) instead of a
# residual-add pass plus y - mean. No effect without lookahead or a residual, or with SEEDED_NORMALIZE.
# Correct, but a measured regression (Refinement 3), so off: the DEST -> srcB move a tile costs more than the
# pack + unpack of the pass it removes (op 8x512 residual, 2-row blocks: 17.4 vs 17.0 us; model 8x512 35.0 vs 34.3;
# model 4x288 14.95 vs 14.8).
FUSED_RECOMPUTE = False


def _lookahead(num_row_blocks):
    return LOOKAHEAD and num_row_blocks > 1


def cb_bytes(block_rows, hidden_tiles, stat_tiles, hidden_split, num_row_blocks=1):
    """L1 the op's own CBs take per core, beyond the io-backed shards (l1_ledger.md § Total)."""
    lookahead = _lookahead(num_row_blocks)
    stats_local_blocks = 2 if lookahead else 1
    return (
        FP32_TILE * block_rows * hidden_tiles  # cb_y
        + FP32_TILE * hidden_tiles  # cb_gamma
        + BF16_TILE * hidden_tiles  # cb_beta
        + FP32_TILE * 2 * block_rows  # cb_row_partials (S then Q partials) | cb_rstd_col
        + FP32_TILE * block_rows  # cb_center_col
        + FP32_TILE * stat_tiles * stats_local_blocks  # cb_stats_local (pair tiles)
        + FP32_TILE * stat_tiles * hidden_split  # cb_stats_gather
    )


def cb_bytes_per_core(plan):
    return cb_bytes(plan.block_rows, plan.hidden_tiles, plan.stat_tiles, plan.hidden_split, plan.num_row_blocks)


def _fp32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _cb(core_ranges, total_size, formats):
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
            for index, dtype, page in formats
        ],
    )


def create_program_descriptor(plan, x, residual, gamma, beta, output, epsilon_bits, compute_config):
    cores = plan.cores
    has_residual = residual is not None
    block_tiles = plan.block_rows * plan.hidden_tiles  # T: pages of one row block
    stat_pages = plan.stat_tiles  # one contributor's pair tiles (S and Q of 2 rows each)

    # ---------------- circular buffers ----------------
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x, core_ranges=cores),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output, core_ranges=cores),
        _cb(cores, FP32_TILE * plan.hidden_tiles, [(CB_GAMMA, ttnn.float32, FP32_TILE)]),
        _cb(cores, BF16_TILE * plan.hidden_tiles, [(CB_BETA, ttnn.bfloat16, BF16_TILE)]),
        _cb(cores, FP32_TILE * block_tiles, [(CB_Y, ttnn.float32, FP32_TILE)]),
        _cb(
            cores,
            FP32_TILE * 2 * plan.block_rows,
            [(CB_ROW_PARTIALS, ttnn.float32, FP32_TILE), (CB_RSTD_COL, ttnn.float32, FP32_TILE)],
        ),
        _cb(
            cores,
            FP32_TILE * plan.block_rows,
            [(CB_CENTER_COL, ttnn.float32, FP32_TILE)],
        ),
    ]
    lookahead = _lookahead(plan.num_row_blocks)
    # With lookahead cb_stats_local holds block b+1's pair tiles while the writer still sends block b's.
    cbs.append(
        _cb(cores, FP32_TILE * stat_pages * (2 if lookahead else 1), [(CB_STATS_LOCAL, ttnn.float32, FP32_TILE)])
    )
    cbs += [
        _cb(cores, FP32_TILE * stat_pages * plan.hidden_split, [(CB_STATS_GATHER, ttnn.float32, FP32_TILE)]),
    ]
    if has_residual:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_RESIDUAL, residual, core_ranges=cores))

    # ---------------- per-core runtime args ----------------
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for index, core in enumerate(plan.core_list):
        hidden_slice = index % plan.hidden_split  # ND shard (b, s, g) sits on core (b*RS + s)*G + g
        reader_rt[core.x][core.y] = [gamma.buffer_address(), beta.buffer_address(), hidden_slice * plan.hidden_tiles]
        group_first = index - hidden_slice
        peers = []
        for member in range(plan.hidden_split):
            peer = x.device().worker_core_from_logical_core(plan.core_list[group_first + member])
            peers.extend([peer.x, peer.y])
        writer_rt[core.x][core.y] = [hidden_slice, *peers]
        compute_rt[core.x][core.y] = [epsilon_bits]

    # ---------------- reader ----------------
    reader_ct = [
        CB_X,
        CB_RESIDUAL,
        CB_GAMMA,
        CB_BETA,
        int(has_residual),
        plan.rows_per_shard * plan.hidden_tiles,  # shard pages (credits pushed once)
        plan.hidden_tiles,
        gamma.buffer_page_size(),
        beta.buffer_page_size(),
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_residual_layer_norm_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
    )

    # ---------------- writer (exchange + output drain) ----------------
    # EXCHANGE_MODE / MODEL_EXCHANGE_MODE select the transport; the concurrent form needs one round per call (slots
    # not reused), so a multi-block plan falls back to the handshaked serialized chain.
    mode = MODEL_EXCHANGE_MODE if plan.placement == "model_layout" else EXCHANGE_MODE
    assert mode in EXCHANGE_MODES
    if mode.startswith("concurrent") and plan.num_row_blocks > 1:
        mode = "serialized_chain"
    exchange_mode = EXCHANGE_MODES.index(mode)
    concurrent_exchange = mode.startswith("concurrent")
    chain_exchange = mode == "serialized_chain"
    writer_ct = [
        CB_STATS_LOCAL,
        CB_STATS_GATHER,
        CB_OUTPUT,
        stat_pages,
        FP32_TILE,
        plan.hidden_split,
        block_tiles,
        plan.num_row_blocks,
        exchange_mode,
        STATS_ARRIVED_SEMAPHORE_ID,
    ]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_residual_layer_norm_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=WRITER_NOC),
    )

    # ---------------- compute ----------------
    hidden_elements = plan.hidden_tiles * plan.hidden_split * 32
    inv_hidden_bits = _fp32_bits(1.0 / hidden_elements)  # fl(1/768) as fp32
    # An odd hidden_tiles (model_layout: 3) cannot be folded in pairwise halves; the row sum then adds every tile to
    # itself (y_j + y_j, exact) and so carries 2*S. Its scale fl(1/768) / 2 is exact, so mean is bit-identical to S *
    # fl(1/768).
    sum_scale = 1 if plan.hidden_tiles % 2 == 0 else 2
    inv_sum_bits = _fp32_bits(1.0 / (hidden_elements * sum_scale))
    compute_ct = [
        plan.block_rows,
        plan.hidden_tiles,
        plan.stat_tiles,
        plan.hidden_split,
        plan.num_row_blocks,
        int(has_residual),
        inv_hidden_bits,
        ELTWISE_BLOCK_TILES,
        plan.stat_slots,
        # CB indices (single source: the constants above; the kernel reads them at namespace scope).
        CB_X,
        CB_RESIDUAL,
        CB_GAMMA,
        CB_BETA,
        CB_Y,
        CB_ROW_PARTIALS,
        CB_STATS_LOCAL,
        CB_STATS_GATHER,
        CB_CENTER_COL,
        CB_RSTD_COL,
        CB_OUTPUT,
        inv_sum_bits,
        int(lookahead),
        int(SEEDED_NORMALIZE),
        int(FUSED_RECOMPUTE),
    ]
    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
    for cb in _unpack_to_dest_fp32_cbs():
        unpack_modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_config.math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=compute_config.math_approx_mode,
    )
    compute_cfg.unpack_to_dest_mode = unpack_modes
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_residual_layer_norm_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_cfg,
    )

    # ---------------- exchange: one fixed-sender channel per group member ----------------
    # Mcast partitions the ordered cores into consecutive groups of hidden_split; channel c's sender is member c
    # of every group. op_layout: row-major triples on the grid may wrap a row (irregular receiver sets): chain
    # unicast. model_layout: column-major order, a group is one grid column (a rectangle).
    descriptor = ttnn.ProgramDescriptor(semaphores=[], cbs=cbs)
    if mode == "concurrent_unicast":
        descriptor.semaphores = [
            ttnn.SemaphoreDescriptor(id=STATS_ARRIVED_SEMAPHORE_ID, core_ranges=cores, initial_value=0)
        ]
    else:
        semaphore_id = MCAST_FIRST_SEMAPHORE_ID
        for member in range(plan.hidden_split):
            channel = ttnn.Mcast(
                x.device(),
                ttnn.McastConfig(
                    noc=WRITER_NOC,
                    handshake=not concurrent_exchange,
                    irregular_receiver_set_mode=(
                        ttnn.TransferMode.ChainUnicast if chain_exchange else ttnn.TransferMode.Multicast
                    ),
                ),
                cores,
                plan.hidden_split,
                ttnn.McastFixedSenderConfig(sender_index=member),
                ttnn.McastCoreOrder.ColumnMajor if plan.column_major else ttnn.McastCoreOrder.RowMajor,
            )
            channel.attach(descriptor, f"{MCAST_PREFIX}{member}", [writer], semaphore_id)
            semaphore_id = channel.next_semaphore_id()
    descriptor.kernels = [reader, writer, compute]
    return descriptor
