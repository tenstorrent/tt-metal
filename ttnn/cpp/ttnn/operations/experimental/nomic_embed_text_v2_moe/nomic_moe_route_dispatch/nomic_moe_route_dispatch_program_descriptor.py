# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_moe_route_dispatch ProgramDescriptor (regime R1 `nd_push_l1`, op_design.md).

Kernels, all on the 11x10 grid (token work only on x's shard cores, tilize on every core):
  * reader  (RISCV_0 / NOC0): x publish, W slice fetch + multicast, partial exchange, selection, count table,
                              metadata writes, its half of the row pushes, owner-side landing count, gathered
                              publish
  * writer  (RISCV_1 / NOC1): the other half of the row pushes; DRAM `dispatched` (R4): the owned blocks to DRAM
  * compute: untilize x, partial logits (matmul_block), logits sum + mask, softmax, tilize of the owned shards
             (in place into the L1 output shard, or streamed through CB_DISPATCHED to the writer for DRAM)
Every non-sharded CB is declared on every core, so a CB has the same L1 address everywhere and peers address
remote CBs by the local address.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from .nomic_moe_route_dispatch import (
    BF16_TILE_BYTES,
    COORDINATOR,
    DRAM_OUT_BLOCKS_IN_FLIGHT,
    FP32_TILE_BYTES,
    GRID_X,
    GRID_Y,
    HIDDEN_TILES,
    NUM_CORES,
    NUM_EXPERTS,
    PUSH_BYTES_IN_FLIGHT,
    PUSH_STAGGER_TOKENS,
    PUSH_CMD_BUFS,
    WEIGHTS_SPREAD,
    TILE,
    TOP_K,
    W_SENDER_TOP,
    W_SLICE_NOCS,
    Plan,
)

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic names; op_design.md Circular Buffers)
CB_X_TILES = 0  # backed on x's shard (nd_norm), or the core's tile rows read from interleaved x (R2)
CB_ROUTER_WEIGHT = 1  # faces 0 / 2 of the 24 W tiles, compact (kernels: w_slot / w_face_tile), 12 tile pages
CB_OWN_PARTIAL = 2  # compute -> reader: this member's partial is packed into its region of CB_PARTIALS (signal)
CB_PARTIALS = 3  # G contributors' partials (contributor-major)
CB_SCORES = 9  # softmax probabilities (fp32; the 8 expert columns)
CB_X_ROWS = 12  # the shard untilized; consumed by the reader, the writer reads its tokens after the slot handshake
CB_SLOT_ROWS = 14  # reader -> writer: every slot row of the core + the region table
CB_GATHERED_ROWS = 15  # row-major rows pushed by the token cores into this core's output blocks
CB_DISPATCHED = 16  # backed on the output shards (L1), or the tilize -> DRAM-writer stream (DRAM)
CB_READER_SCRATCH = 17  # count table (multicast target), drain word, expert_counts staging
CB_SELECT_GO = 18  # reader -> writer: scores are in, the slot page and weight tiles addresses
CB_WRITER_COUNTS = 19  # writer -> reader: per-expert counts of the writer's tokens
CB_WEIGHT_TILES = 20  # reader scratch: the leader's combine_weights tiles (column 0 = weight, rest 0)
CB_STAGE_TILES = 21  # whole-piece push: the staged tile range (own + peers' tiles), untilized into cb_x_rows
CB_OWN_PARTIAL_WEST = 22  # compute -> writer: the same signal (the writer sends to the lower members on NOC1)
CB_W_FACES = 23  # 1 KB-page view of CB_ROUTER_WEIGHT: the matmul's in1 (faces-only tiles sit on 1 KB boundaries)
CB_CODE_STAGE = 24  # compute -> DM, one code tile a pass (its face goes to the group's cb_scores)
CB_CODE_FLAGS = 25  # per-member landed-pass counters + flag-write source constants
CODE_FLAGS_BYTES = 16 * (3 + 8)  # kMaxSlices counters + up to 2 * MAX_ROWS_PER_SHARD pass constants, 16 B each
W_FACE_BYTES = 1024
W_CB_TILES = HIDDEN_TILES // 2  # faces 0 and 2 of every W tile
HANDSHAKE_PAGE_BYTES = 16
NUM_CIRCULAR_BUFFERS = 64


def max_passes(p) -> int:
    """Softmax passes (one 16-token half each) the busiest member of a group runs."""
    return 1 if p.hidden_split == 1 else -(-2 * p.rows_per_shard // p.hidden_split)


# Partial tiles of contributor c landed on this member (one counter a contributor, written by c's writer)
SEM_PARTIALS = (0, 1, 2)
SEM_W_LOCAL = 3  # a W sender core: the RISC off the channel's NoC has read its half of the channel's DRAM reads
MAX_SEMAPHORES = 16

L1_ALIGN = 64  # scratch sections start on a DRAM-write-compatible boundary
TABLE_ROW_BYTES = 16  # 8 u16 counts
# slot-rows header: starts[8], totals[8], regions_end, staging address, reader counts[2], table address, table flag,
# two table-scan blocks of 9 words (kernels: kHeader*)
REGION_WORDS = 40

READER_NOC = ttnn.NOC.NOC_0
WRITER_NOC = ttnn.NOC.NOC_1


# Debug scaffolding (ablation profiling, /perf-measure): NOMIC_MOE_ROUTE_DISPATCH_ABLATE="A,B" compiles the kernels
# with -DABLATE_A -DABLATE_B; a name in KERNEL_LIB_ABLATIONS (the helpers' own payload switches) is passed as is.
# Outputs are wrong under any ablation; never set in production.
KERNEL_LIB_ABLATIONS = ("SKIP_COMPUTE", "CKL_ELTWISE_CHAIN_SKIP_COMPUTE")


def _ablation_defines():
    names = [n.strip() for n in os.environ.get("NOMIC_MOE_ROUTE_DISPATCH_ABLATE", "").split(",") if n.strip()]
    return [(n if n in KERNEL_LIB_ABLATIONS else f"ABLATE_{n}", "1") for n in names]


def _dm_defines():
    """Data-movement kernel defines: the ablation scaffolding."""
    return _ablation_defines()


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def slot_rows_bytes(p: Plan) -> int:
    return TOP_K * p.rows_per_shard * TILE * 4 + REGION_WORDS * 4


def scratch_layout(p: Plan) -> tuple[int, int, int]:
    """(drain offset, expert_counts staging offset, total bytes) of the reader scratch; the table is at 0."""
    drain = _round_up(p.num_groups * TABLE_ROW_BYTES, L1_ALIGN)
    counts = drain + L1_ALIGN
    return drain, counts, counts + L1_ALIGN


def _cb_specs(p: Plan):
    """(index, dtype, page bytes, pages) of every CB the op allocates (the sharded ones excluded)."""
    rps, ht, w = p.rows_per_shard, p.hidden_tiles, p.block_width_tiles
    fp32, bf16 = ttnn.float32, ttnn.bfloat16
    return (
        [
            (CB_ROUTER_WEIGHT, fp32, FP32_TILE_BYTES, W_CB_TILES),
            (CB_OWN_PARTIAL, fp32, HANDSHAKE_PAGE_BYTES, rps),
            (CB_OWN_PARTIAL_WEST, fp32, HANDSHAKE_PAGE_BYTES, rps),
            (CB_PARTIALS, fp32, FP32_TILE_BYTES, p.hidden_split * rps),
            (CB_WEIGHT_TILES, fp32, FP32_TILE_BYTES, TOP_K * rps),
            (CB_SCORES, fp32, FP32_TILE_BYTES, rps),
            (CB_CODE_STAGE, fp32, FP32_TILE_BYTES, max_passes(p)),
            (CB_CODE_FLAGS, ttnn.uint32, CODE_FLAGS_BYTES, 1),
            (CB_X_ROWS, bf16, BF16_TILE_BYTES, rps * p.stage_tiles),
            (CB_SLOT_ROWS, ttnn.uint32, slot_rows_bytes(p), 1),
            (CB_GATHERED_ROWS, bf16, BF16_TILE_BYTES, p.max_blocks_per_core * w),
            (CB_READER_SCRATCH, ttnn.uint32, scratch_layout(p)[2], 1),
            (CB_SELECT_GO, ttnn.uint32, HANDSHAKE_PAGE_BYTES, 1),
            (CB_WRITER_COUNTS, ttnn.uint32, HANDSHAKE_PAGE_BYTES, 1),
        ]
        + ([(CB_DISPATCHED, bf16, BF16_TILE_BYTES, DRAM_OUT_BLOCKS_IN_FLIGHT * w)] if p.out_dram else [])
        + ([(CB_STAGE_TILES, bf16, BF16_TILE_BYTES, rps * p.stage_tiles)] if p.push_gather else [])
        + (
            # Exactly rps*ht pages: the reader re-publishes the same pages for the matmul round after the untilize pops.
            [(CB_X_TILES, bf16, BF16_TILE_BYTES, rps * ht)]
            if p.x_interleaved
            else []
        )
    )


def _push_args(p: Plan, member) -> list:
    """RT args of a pushing RISC (kernels: RowPusher): first staged tile, piece count, then k pieces of
    (piece, src offset, dst offset, bytes, rows), zero padded."""
    lo, parts = p.member_push[member] if member is not None else (0, ())
    words = [lo, len(parts)]
    for part in parts:
        words.extend(part)
    return words + [0] * (2 + 5 * p.pieces_per_row - len(words))


def cb_bytes(p: Plan) -> int:
    return sum(page * pages for _, _, page, pages in _cb_specs(p))


def _grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])


def _logical(core_idx):
    return ttnn.CoreCoord(core_idx % GRID_X, core_idx // GRID_X)


def _noc_coords(device):
    coords = []
    for core_idx in range(NUM_CORES):
        virtual = device.worker_core_from_logical_core(_logical(core_idx))
        coords.append((virtual.x << 16) | virtual.y)
    return coords


def _compute_config(config, approx):
    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
    unpack_modes[CB_PARTIALS] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=config.math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=approx,
    )
    compute.unpack_to_dest_mode = unpack_modes
    return compute


def create_program_descriptor(
    p: Plan, x, weight, dispatched, combine_index, combine_weights, expert_counts, compute_kernel_config
):
    device = x.device()
    grid = _grid()
    G = p.hidden_split
    if p.x_interleaved:
        token_core_of = {core: core for core in range(p.num_token_cores)}  # R2: token core j = grid core j
    else:
        x_cores = ttnn.corerange_to_cores(x.memory_config().nd_shard_spec.grid, None, True)
        grid_index = {(c.x, c.y): c.y * GRID_X + c.x for c in ttnn.corerange_to_cores(grid, None, True)}
        token_core_of = {}  # grid index -> token core index (shard index)
        for shard, core in enumerate(x_cores):
            token_core_of[grid_index[(core.x, core.y)]] = shard

    # ---------------- circular buffers ----------------
    cbs = [
        ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
            + (
                [ttnn.CBFormatDescriptor(buffer_index=CB_W_FACES, data_format=dtype, page_size=W_FACE_BYTES)]
                if index == CB_ROUTER_WEIGHT
                else []
            ),
        )
        for index, dtype, page, pages in _cb_specs(p)
    ]
    if not p.x_interleaved:
        x_core_ranges = x.memory_config().nd_shard_spec.grid
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X_TILES, x, core_ranges=x_core_ranges))
    if not p.out_dram:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_DISPATCHED, dispatched, core_ranges=grid))

    semaphores = [
        ttnn.SemaphoreDescriptor(id=sem, core_ranges=grid, initial_value=0) for sem in [*SEM_PARTIALS, SEM_W_LOCAL]
    ]

    # ---------------- per-core roles ----------------
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    # Token cores: the x shard map (one shard a core). Group g = shard // G, member c = shard % G.
    token_grid_idx = [0] * p.num_token_cores
    for core_idx, shard in token_core_of.items():
        token_grid_idx[shard] = core_idx
    # Kernels address shard j's core as grid core j (the layer norm places shards on the first cores row-major).
    assert token_grid_idx == list(range(p.num_token_cores)), "x shard j must sit on grid core j"
    for core_idx in range(NUM_CORES):
        logical = _logical(core_idx)
        shard = token_core_of.get(core_idx)
        is_token = int(shard is not None)
        member, group = (shard % G, shard // G) if is_token else (0, 0)
        first_token, core_tokens = p.group_tokens(group) if is_token else (0, 0)
        blocks = p.blocks_of(core_idx)
        reader_tokens = p.reader_tokens_of(group) if is_token else 0
        push_args = _push_args(p, member) if is_token else _push_args(p, None)
        reader_rt[logical.x][logical.y] = [
            core_idx,
            is_token,
            member,
            group,
            first_token,
            core_tokens,
            blocks,
            x.buffer_address(),
            x.buffer_page_size(),
            reader_tokens,
            int(is_token and (WEIGHTS_SPREAD or member == 0)),
            *push_args,
        ]
        writer_rt[logical.x][logical.y] = [
            is_token,
            member,
            first_token,
            core_tokens,
            combine_index.buffer_address(),
            combine_index.buffer_page_size(),
            core_idx,
            blocks,
            dispatched.buffer_address(),
            dispatched.buffer_page_size(),
            reader_tokens,
            group,
            x.buffer_address(),
            int(is_token and (WEIGHTS_SPREAD or member == 0)),
            *push_args,
        ]
        stages = int(is_token and len(p.member_push[member][1]) > 0)
        compute_rt[logical.x][logical.y] = [is_token, member, blocks, stages, reader_tokens, core_tokens]

    coords = _noc_coords(device)

    common_named = [
        ("hidden_tiles", p.hidden_tiles),
        ("rows_per_shard", p.rows_per_shard),
        ("hidden_split", G),
        ("push_bytes_in_flight", PUSH_BYTES_IN_FLIGHT),
        ("push_stagger_tokens", PUSH_STAGGER_TOKENS),
        ("push_cmd_bufs", PUSH_CMD_BUFS),
        ("push_gather", int(p.push_gather)),
        ("stage_tiles", p.stage_tiles),
        ("cb_stage_tiles", CB_STAGE_TILES),
        ("block_width_tiles", p.block_width_tiles),
        ("pieces_per_row", p.pieces_per_row),
        ("num_cores", NUM_CORES),
        ("coordinator", COORDINATOR),
        ("num_groups", p.num_groups),
        ("rows", p.rows),
        ("num_experts", NUM_EXPERTS),
        ("top_k", TOP_K),
        ("cb_x_rows", CB_X_ROWS),
        ("cb_slot_rows", CB_SLOT_ROWS),
        ("cb_gathered_rows", CB_GATHERED_ROWS),
        ("cb_scores", CB_SCORES),
        ("cb_select_go", CB_SELECT_GO),
        ("cb_writer_counts", CB_WRITER_COUNTS),
        ("region_words", REGION_WORDS),
        ("handshake_page_bytes", HANDSHAKE_PAGE_BYTES),
        ("cb_router_weight", CB_ROUTER_WEIGHT),
        ("weight_tiles", W_CB_TILES),
        ("w_sender_top", W_SENDER_TOP),
        ("w_slice_nocs", W_SLICE_NOCS),
        ("sem_w_local", SEM_W_LOCAL),
        ("cb_code_stage", CB_CODE_STAGE),
        ("cb_code_flags", CB_CODE_FLAGS),
    ]
    drain_offset, counts_offset, _ = scratch_layout(p)
    reader_named = common_named + [
        ("cb_x_tiles", CB_X_TILES),
        ("cb_own_partial", CB_OWN_PARTIAL),
        ("cb_partials", CB_PARTIALS),
        ("cb_reader_scratch", CB_READER_SCRATCH),
        ("cb_weight_tiles", CB_WEIGHT_TILES),
        ("weights_stride", G if WEIGHTS_SPREAD else 1),
        ("t_pad", p.t_pad),
        ("drain_offset", drain_offset),
        ("counts_offset", counts_offset),
        ("x_interleaved", int(p.x_interleaved)),
        ("table_row_bytes", TABLE_ROW_BYTES),
        *[(f"sem_partials{c}", sem) for c, sem in enumerate(SEM_PARTIALS)],
    ]
    reader_ct = []
    for tensor in (weight, combine_index, combine_weights, expert_counts, x):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    addresses = [t.buffer_address() for t in (weight, combine_index, combine_weights, expert_counts)]
    page_sizes = [t.buffer_page_size() for t in (weight, combine_index, combine_weights, expert_counts)]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_moe_route_dispatch_reader.cpp"),
        core_ranges=grid,
        compile_time_args=reader_ct,
        named_compile_time_args=reader_named,
        runtime_args=reader_rt,
        common_runtime_args=addresses + page_sizes + coords,
        defines=_dm_defines(),
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=READER_NOC),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_moe_route_dispatch_writer.cpp"),
        core_ranges=grid,
        compile_time_args=ttnn.TensorAccessorArgs(combine_index).get_compile_time_args()
        + ttnn.TensorAccessorArgs(dispatched).get_compile_time_args()
        + ttnn.TensorAccessorArgs(weight).get_compile_time_args()
        + ttnn.TensorAccessorArgs(combine_weights).get_compile_time_args(),
        named_compile_time_args=common_named
        + [("out_dram", int(p.out_dram)), ("cb_dispatched", CB_DISPATCHED), ("hidden_tiles_total", HIDDEN_TILES)]
        + [("weights_stride", G if WEIGHTS_SPREAD else 1), ("t_pad", p.t_pad), ("cb_weight_tiles", CB_WEIGHT_TILES)]
        + [("cb_own_partial_west", CB_OWN_PARTIAL_WEST), ("cb_partials", CB_PARTIALS)]
        + [(f"sem_partials{c}", sem) for c, sem in enumerate(SEM_PARTIALS)],
        runtime_args=writer_rt,
        common_runtime_args=coords
        + [weight.buffer_address(), weight.buffer_page_size()]
        + [combine_weights.buffer_address(), combine_weights.buffer_page_size()],
        defines=_dm_defines(),
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=WRITER_NOC),
    )
    compute_named = [
        ("hidden_tiles", p.hidden_tiles),
        ("rows_per_shard", p.rows_per_shard),
        ("hidden_split", G),
        ("block_width_tiles", p.block_width_tiles),
        ("weight_tiles", W_CB_TILES),
        ("w_slice_nocs", W_SLICE_NOCS),
        ("cb_x_tiles", CB_X_TILES),
        ("cb_router_weight", CB_ROUTER_WEIGHT),
        ("cb_own_partial", CB_OWN_PARTIAL),
        ("cb_own_partial_west", CB_OWN_PARTIAL_WEST),
        ("cb_w_faces", CB_W_FACES),
        ("cb_partials", CB_PARTIALS),
        ("cb_scores", CB_SCORES),
        ("cb_x_rows", CB_X_ROWS),
        ("push_gather", int(p.push_gather)),
        ("stage_tiles", p.stage_tiles),
        ("cb_stage_tiles", CB_STAGE_TILES),
        ("cb_gathered_rows", CB_GATHERED_ROWS),
        ("cb_dispatched", CB_DISPATCHED),
        ("cb_code_stage", CB_CODE_STAGE),
    ]
    approx = bool(compute_kernel_config.math_approx_mode)
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_moe_route_dispatch_compute.cpp"),
        core_ranges=grid,
        compile_time_args=[],
        named_compile_time_args=compute_named,
        runtime_args=compute_rt,
        defines=[("EXP_APPROX", "1" if approx else "0")] + _ablation_defines(),
        config=_compute_config(compute_kernel_config, approx),
    )

    # ---------------- multicast channels (W slices, count table) ----------------
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)
    semaphore_id = SEM_W_LOCAL + 1
    # (prefix, sender core, sent by the writer): one W channel a NoC, carrying every slice c with c % W_SLICE_NOCS
    # equal to its index (odd slices on the writer's NoC).
    channels = [(f"w{c}", W_SENDER_TOP - c, c % W_SLICE_NOCS == 1) for c in range(min(G, W_SLICE_NOCS))]
    channels += [("table", COORDINATOR, False)]
    for prefix, sender, by_writer in channels:
        # A writer-sent slice's sender must not hold tokens: its writer would wait for its own (source-excluded) multicast.
        assert not by_writer or sender not in token_core_of, "writer-sent W slice on a token core"
        channel = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=WRITER_NOC if by_writer else READER_NOC, handshake=False),
            grid,
            NUM_CORES,
            ttnn.McastFixedSenderConfig(sender_index=sender),
            ttnn.McastCoreOrder.RowMajor,
        )
        channel.attach(descriptor, prefix, [writer] if by_writer else [reader], semaphore_id)
        semaphore_id = channel.next_semaphore_id()
    assert semaphore_id <= MAX_SEMAPHORES, f"{semaphore_id} semaphores a core"
    descriptor.kernels = [reader, writer, compute]
    return descriptor
