# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm_head_rope — ProgramDescriptor (CBs, kernels, per-core args).

Blocking model (op_design.md): tile space = row-tile r x head h x head-dim tile d (4).
A block is 1 row-tile x block_heads heads x HEAD_TILES tiles, owned by one core.
Cores split (row-group, head-group) = (Rg, G), chosen by select_grid_split(). Each core's first row-tile is static
(its row-group index); the remaining row-tiles of a head group are claimed dynamically by its Rg cores through a
fetch-and-add counter (SEM_ROW_COUNTER on the group's first core) — see the reader kernel header for why.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------- fixed geometry
TILE_DIM = 32
HEAD_DIM = 128
HEAD_TILES = HEAD_DIM // TILE_DIM  # head-dim tiles per head (dependent axis, never split)

# ---------------------------------------------------------------- blocking knobs (single source of truth)
MAX_BLOCK_HEADS = 8  # L1 working-set cap on block_heads (measured flat over {1,2,4,8} at 2304x3072)
MAX_RESIDENT_HEADS = 24  # L1 cap on heads_per_core (resident weight)
BLOCK_ROW_TILES = 1  # row-tiles per block (Phase 0: 1; raising it needs strided cos/sin addressing)
INPUT_DEPTH = 2  # cb_input_tiles depth, in blocks. Perf 1 re-measured with dynamic row claiming (median of 3):
#                  depth 1 vs 2 — 2048x3072 plain 80.1 vs 74.2 us, weight-only 84.5 vs 76.5, w+rope 2304 91.5 vs 93.3
#                  (noise band); 2 everywhere.
OUTPUT_DEPTH = 2  # cb_output_tiles depth, in blocks
ROPE_DEPTH = 2  # cb_rope_cos / cb_rope_sin depth, in row-tiles
STORE_CHUNK_HEADS = 1  # writer streams the output block in chunks of this many heads (flush per chunk)
READ_CHUNK_TILES = 8  # reader barriers every this many x tiles (caps a core's outstanding reads; ~-3.5% measured)
NOC1_READ_ROWS = 2  # logical core rows [0, NOC1_READ_ROWS) read on NOC_1 / write on NOC_0 (swapped). Blackhole NoC0
#                     serves DRAM reads in core-y order and starves the top rows (~60 us to first byte at 120 cores)
NOC1_MIN_GRID_ROWS = 8  # carve-out: no swap on grids using fewer core rows (measured regression, see below)
SEM_PEER_READY = 1  # cos/sin team: on a team leader, peers +1 when their x/cos/sin space for the next row is reserved
SEM_ROW_MAILBOX = 2  # cos/sin team: on a peer, leader writes the claimed row + 1 (or 0xFFFFFFFF end marker)
SEM_ROPE_VALID = 3  # cos/sin team: on a peer, leader sets 1 after cos/sin of the current row landed in its CB slot
COSSIN_TEAM_MAX = 4  # cos/sin team size cap (consecutive head groups of a row-group). Measured (cossin_share): a full
#                      team of 24 (32x3072) regressed +33% (leader fan-out/coupling); cap 3/4 wins on every rope cell.
SEM_ROW_COUNTER = 0  # program semaphore id: per-head-group dynamic row counter (lives on the group's first core)

# ---------------------------------------------------------------- CB indices (semantic names)
CB_INPUT_TILES = 0
CB_ROPE_COS = 1
CB_ROPE_SIN = 2
CB_WEIGHT_ROW = 3
CB_TRANS_MAT = 4
CB_COLSEL = 5  # S_h column-select tiles for the packed-column rsqrt (writer-generated, resident)
CB_WEIGHT = 6
CB_SUMSQ = 7
CB_RSTD = 8
CB_NORMED = 9
CB_COLEXT = 10  # E_h column-extract tiles for the packed-column rsqrt (writer-generated, resident)
CB_RSTD_PACKED = 11  # one DEST-width tile: every head's rstd in its own column
CB_OUTPUT_TILES = 16
CB_CTRL_COMPUTE = 12  # reader -> compute: claimed row-tile index / end marker (one uint32 per page)
CB_CTRL_WRITER = 13  # reader -> writer: same word (separate CB: single-consumer invariant)
CB_W_STAGE = 14  # [weight forwarding] compact weight row-0 bytes, 128 B per weight tile (face0 | face1 rows 0-1)
W_STAGE_BYTES_PER_TILE = 128
SEM_W_READY = 4  # [weight forwarding] peer's "staging filled by its leader" flag (set by a plain remote write)
W_MAX_PEERS = 20  # [weight forwarding] peers per leader: bounds one leader's push to <= 20 x heads_per_core x 512 B.
#                   Perf 2 (3840x3072 weight-only, G=1, Rg=120): 1 leader 135 us, <=40 peers 124 us, <=20 peers 121 us.
CTRL_PAGE_BYTES = 32
CTRL_DEPTH = 2


# ---------------------------------------------------------------- work-split cost model (Perf 3, split_granularity)
# Units: head-rows (1 head x 1 row-tile = 4 tiles in + 4 out). Constants measured on Blackhole, 120 cores.
BLOCK_OVERHEAD_HEADS = 0.15  # fixed per-block cost (CB handshakes, read barrier, flush, compute block init), in
#                              head-row units. Measured 2048x3072 plain, G=3 Rg=40: block_heads 1/2/4/8 =
#                              75.2/70.8/67.0/67.0 us -> ~0.6 us per extra block vs ~4.2 us per head-row.
TAIL_WEIGHT_DM = 0.25  # cost of a core's work above the mean when rows are DRAM-bound (no rope): the tail runs with
#                        fewer contenders, so it drains fast. Measured: 2048x3072 plain G=2 Rg=60 (max 24 head-rows vs
#                        mean 12.8) 64.5-66.0 us, same as balanced splits.
WEIGHT_FWD_HEADS = 0.5  # fixed start-up cost of weight forwarding (dynamic grids only: leader's DRAM read + hop to
#                         its peers before their first block can finish). Measured 1024x1024 weight-only: static (3, 32)
#                         14.07 us vs dynamic (8, 15) 14.77 us although the rest of the model favours (8, 15) by ~9%.
TAIL_WEIGHT_ROPE = 1.0  # rope rows carry ~2.6 us/head-row of compute, which the tail cannot hide: full cost.
#                         Measured 2304x3072 w+rope: max head-rows 18/20/21/26 -> 75/84.5/84.7/87.7 us.


def split_cost(Rt, H, G, Rg, has_rope, has_weight=False):
    """Modelled kernel time of a (G, Rg) split, in head-row units (1 head x 1 row-tile = 4 tiles in + 4 out)."""
    hpc = -(-H // G)
    rpc = -(-Rt // Rg)
    bh = min(hpc, MAX_BLOCK_HEADS)
    blocks_per_row = -(-hpc // bh)
    rope = 1 if has_rope else 0
    teams = -(-G // min(G, COSSIN_TEAM_MAX)) if has_rope else 0
    # mean per-core work: x/out traffic + cos/sin (read once per team per row) + per-block fixed costs
    u_avg = (Rt * H + Rt * teams * rope) / (G * Rg) * (1 + BLOCK_OVERHEAD_HEADS * blocks_per_row / hpc)
    # busiest core: claim quantization (rpc rows) x heads, +1 per row for the rope (cos/sin wait + rope work)
    u_max = rpc * (hpc + rope)
    lam = TAIL_WEIGHT_ROPE if has_rope else TAIL_WEIGHT_DM
    fixed = WEIGHT_FWD_HEADS if (has_weight and Rg < Rt) else 0.0
    return u_avg + lam * max(0.0, u_max - u_avg) + fixed


def _static_split(Rt, H, num_cores, has_rope):
    """Busiest-core rule (minimize per-core units, then total, then cores). Used to pick the static (Rg == Rt) split."""
    best = None
    for G in range(1, min(H, num_cores) + 1):
        Rg = min(Rt, num_cores // G)
        if Rg < 1:
            continue
        hpc = -(-H // G)
        rpc = -(-Rt // Rg)
        if hpc > MAX_RESIDENT_HEADS:
            continue
        per_core = rpc * (hpc + (1 if has_rope else 0))
        total = Rt * H + (Rt * G if has_rope else 0)
        key = (per_core, total, G * Rg)
        if best is None or key < best[0]:
            best = (key, G, Rg)
    if best is None:
        # Only reachable when H > MAX_RESIDENT_HEADS * num_cores (W > ~368K on a 120-core grid).
        raise ValueError(
            f"rms_norm_head_rope: {H} heads cannot be split so that each core holds <= {MAX_RESIDENT_HEADS} "
            f"heads on {num_cores} cores (resident-weight L1 cap)"
        )
    return best[1], best[2]


def select_grid_split(Rt, H, num_cores, has_rope, has_weight=False):
    """Pick (G head-groups, Rg row-groups) minimizing split_cost. Candidates: every dynamic split (Rg < Rt, rows
    claimed) plus the busiest-core rule's split (_static_split; static when Rg == Rt). Ties -> less DRAM traffic,
    then fewer cores. Deterministic."""
    cands = []
    for G in range(1, min(H, num_cores) + 1):
        Rg = min(Rt, num_cores // G)
        if Rg < 1 or Rg >= Rt or -(-H // G) > MAX_RESIDENT_HEADS:
            continue
        cands.append((G, Rg))
    cands.append(_static_split(Rt, H, num_cores, has_rope))  # raises when no split respects MAX_RESIDENT_HEADS
    best = None
    for G, Rg in cands:
        teams = -(-G // min(G, COSSIN_TEAM_MAX)) if has_rope else 0
        key = (round(split_cost(Rt, H, G, Rg, has_rope, has_weight), 6), Rt * H + Rt * teams, G * Rg)
        if best is None or key < best[0]:
            best = (key, G, Rg)
    return best[1], best[2]


def _split(n, parts):
    """First n % parts parts get ceil, the rest floor. Returns list of (start, count)."""
    base, rem = divmod(n, parts)
    out, start = [], 0
    for i in range(parts):
        c = base + (1 if i < rem else 0)
        out.append((start, c))
        start += c
    return out


def _core_ranges(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _cb(index, num_pages, page_size, fmt, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)],
    )


def _float_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def create_program_descriptor(input_tensor, output_tensor, weight, rope_cos, rope_sin, epsilon, compute_config):
    assert BLOCK_ROW_TILES == 1, "Phase 0: block_row_tiles must be 1 (cos/sin Row mapping)"
    has_weight = weight is not None
    has_rope = rope_cos is not None

    R, W = input_tensor.shape[-2], input_tensor.shape[-1]
    Rt = R // TILE_DIM
    Wt = W // TILE_DIM
    H = W // HEAD_DIM

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    num_cores_avail = grid.x * grid.y

    G, Rg = select_grid_split(Rt, H, num_cores_avail, has_rope, has_weight)
    heads_per_core = -(-H // G)  # HC: max heads over cores
    block_heads = min(heads_per_core, MAX_BLOCK_HEADS)
    block_tiles = block_heads * HEAD_TILES  # BT

    head_split = _split(H, G)
    # Rows beyond each core's static first row exist only when Rg < Rt; otherwise the claim + ctrl handshake is
    # compiled out of all three kernels (carve-out, measured +3-4% on such grids).
    dynamic_rows = 1 if Rg < Rt else 0
    # Weight forwarding (Perf 2): the reader's weight-first prologue made every core read the row-0 bytes of its
    # head group's weight tiles itself (Rg x redundant 64 B DRAM reads, all at kernel start, worst cores 60 us).
    # Instead the group's leader(s) read them once, on the writer RISC, and push a compact copy to their peers.
    # Carve-out (measured): only with dynamic rows. With one static row per core (Rg >= Rt, short kernels) it
    # regressed 256x3072 weight-only +27%, 512x3072 w+rope +7.6%, 1024x1024 weight-only +14% (synchronized
    # compute start -> bursty writes), so those grids keep the reader prologue.
    weight_fwd = has_weight and dynamic_rows == 1
    num_cores = G * Rg
    cores = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(num_cores)]  # row-wise enumeration
    # Dynamic rows: group g's counter lives on its first core (k = g); it starts at Rg because each core's first
    # row-tile (its row-group index rg < Rg <= Rt) is static.
    counter_cores = [device.worker_core_from_logical_core(cores[g]) for g in range(G)]
    core_ranges = _core_ranges(cores)
    # cos/sin teams: head groups of a row-group are chunked into consecutive teams of <= COSSIN_TEAM_MAX. A team
    # processes the same row-tiles; its leader claims them (one counter per team index t, on team t's leader in
    # row-group 0), reads cos/sin once and forwards them to its peers. Team size 1 == per-head-group claiming.
    T = min(G, COSSIN_TEAM_MAX)
    team = has_rope and T > 1
    team_of = lambda g: (g // T) * T  # first head group of g's team
    team_len = lambda g: min(G, team_of(g) + T) - team_of(g)
    if team:
        counter_cores = [counter_cores[team_of(g)] for g in range(G)]
    vcores = [device.worker_core_from_logical_core(c) for c in cores]

    # ---------------------------------------------------------------- CBs
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    dest_fmt = ttnn.float32 if compute_config.fp32_dest_acc_en else ttnn.bfloat16
    dest_tile = ttnn.tile_size(dest_fmt)

    cbs = [
        _cb(CB_INPUT_TILES, INPUT_DEPTH * block_tiles, bf16_tile, ttnn.bfloat16, core_ranges),
        _cb(CB_COLSEL, block_heads, bf16_tile, ttnn.bfloat16, core_ranges),
        _cb(CB_COLEXT, block_heads, bf16_tile, ttnn.bfloat16, core_ranges),
        _cb(CB_RSTD_PACKED, 1, dest_tile, dest_fmt, core_ranges),
        _cb(CB_SUMSQ, block_heads, dest_tile, dest_fmt, core_ranges),
        _cb(CB_RSTD, block_heads, dest_tile, dest_fmt, core_ranges),
        _cb(CB_OUTPUT_TILES, OUTPUT_DEPTH * block_tiles, bf16_tile, ttnn.bfloat16, core_ranges),
        _cb(CB_CTRL_COMPUTE, CTRL_DEPTH, CTRL_PAGE_BYTES, ttnn.uint32, core_ranges),
        _cb(CB_CTRL_WRITER, CTRL_DEPTH, CTRL_PAGE_BYTES, ttnn.uint32, core_ranges),
    ]
    if has_weight:
        cbs += [
            _cb(CB_WEIGHT_ROW, heads_per_core * HEAD_TILES, bf16_tile, ttnn.bfloat16, core_ranges),
            _cb(CB_WEIGHT, heads_per_core * HEAD_TILES, bf16_tile, ttnn.bfloat16, core_ranges),
        ]
        if weight_fwd:
            cbs.append(_cb(CB_W_STAGE, heads_per_core * HEAD_TILES, W_STAGE_BYTES_PER_TILE, ttnn.bfloat16, core_ranges))
    if has_rope:
        cbs += [
            _cb(CB_ROPE_COS, ROPE_DEPTH * HEAD_TILES, bf16_tile, ttnn.bfloat16, core_ranges),
            _cb(CB_ROPE_SIN, ROPE_DEPTH * HEAD_TILES, bf16_tile, ttnn.bfloat16, core_ranges),
            _cb(CB_TRANS_MAT, 1, bf16_tile, ttnn.bfloat16, core_ranges),
            _cb(CB_NORMED, block_tiles, dest_tile, dest_fmt, core_ranges),
        ]

    defines = []
    if has_weight:
        defines.append(("HAS_WEIGHT", "1"))
    if has_rope:
        defines.append(("HAS_ROPE", "1"))
    if team:
        defines.append(("COSSIN_TEAM", "1"))
    if weight_fwd:
        defines.append(("WEIGHT_FWD", "1"))

    # ---------------------------------------------------------------- CT args
    def ta(t):
        # Absent optional tensors keep their CT-arg slot (kernels chain offsets unconditionally); the input's
        # accessor args are a shape-valid placeholder that the kernel never dereferences.
        return ttnn.TensorAccessorArgs(t if t is not None else input_tensor).get_compile_time_args()

    reader_ct = [
        CB_INPUT_TILES,
        CB_ROPE_COS,
        CB_ROPE_SIN,
        CB_WEIGHT_ROW,
        CB_TRANS_MAT,
        block_heads,
        HEAD_TILES,
        Wt,
        READ_CHUNK_TILES,
        CB_CTRL_COMPUTE,
        CB_CTRL_WRITER,
        SEM_ROW_COUNTER,
        dynamic_rows,
        SEM_PEER_READY,
        SEM_ROW_MAILBOX,
        SEM_ROPE_VALID,
    ]
    reader_ct += ta(input_tensor) + ta(rope_cos) + ta(rope_sin) + ta(weight)

    writer_ct = [
        CB_OUTPUT_TILES,
        block_heads,
        HEAD_TILES,
        Wt,
        STORE_CHUNK_HEADS,
        CB_COLSEL,
        CB_COLEXT,
        CB_CTRL_WRITER,
        dynamic_rows,
        CB_WEIGHT_ROW,
        CB_W_STAGE,
        SEM_W_READY,
    ]
    writer_ct += ta(output_tensor) + ta(weight)

    compute_ct = [
        CB_INPUT_TILES,
        CB_ROPE_COS,
        CB_ROPE_SIN,
        CB_WEIGHT_ROW,
        CB_TRANS_MAT,
        CB_COLSEL,
        CB_WEIGHT,
        CB_SUMSQ,
        CB_RSTD,
        CB_NORMED,
        CB_COLEXT,
        CB_OUTPUT_TILES,
        block_heads,
        HEAD_TILES,
        CB_RSTD_PACKED,
        CB_CTRL_COMPUTE,
        dynamic_rows,
    ]

    # ---------------------------------------------------------------- RT args (per core)
    reader_rt = {}
    writer_rt = {}
    compute_rt = ttnn.RuntimeArgs()
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    w_addr = weight.buffer_address() if has_weight else 0
    cos_addr = rope_cos.buffer_address() if has_rope else 0
    sin_addr = rope_sin.buffer_address() if has_rope else 0
    eps_bits = _float_bits(epsilon)

    # Weight forwarding leaders: the group's last L cores (highest rg = bottom grid rows, never NoC-swapped, so the
    # writer reads DRAM on NOC_1; a leader on a swapped top row reads on the starved NOC_0 and measured 50 us).
    # The other cores are dealt round-robin to them, <= W_MAX_PEERS each.
    w_leaders = max(1, -(-(Rg - 1) // W_MAX_PEERS))
    for k, core in enumerate(cores):
        rg, g = divmod(k, G)
        head_start, num_heads = head_split[g]
        reader_rt[(core.x, core.y)] = [
            x_addr,
            cos_addr,
            sin_addr,
            w_addr,
            rg,  # static first row-tile
            Rt,  # rows per head group (claim limit)
            head_start,
            num_heads,
            counter_cores[g].x,
            counter_cores[g].y,
        ]
        if team:
            t0, tn = team_of(g), team_len(g)
            lead = vcores[rg * G + t0]
            peers = vcores[rg * G + t0 + 1 : rg * G + t0 + tn]
            # mcast (mode 1) when the peers are the cores right of the leader in its grid row, else unicast (mode 0)
            peer_row = all(p.y == lead.y for p in peers) and all(
                peers[i].x == lead.x + 1 + i for i in range(len(peers))
            )
            mode = 1 if peers and peer_row else 0
            rect = (peers[0].x, peers[0].y, peers[-1].x, peers[-1].y) if peers else (0, 0, 0, 0)
            reader_rt[(core.x, core.y)] += [
                1 if g == t0 else 0,  # is leader
                lead.x,
                lead.y,
                tn - 1,  # num peers
                mode,
                *rect,  # mcast rectangle, noc0 orientation (start <= end)
            ]
            if g == t0:
                for p in peers:
                    reader_rt[(core.x, core.y)] += [p.x, p.y]
        peers = []
        is_leader = 1
        if weight_fwd:
            first_leader = Rg - w_leaders
            my_leader = rg if rg >= first_leader else first_leader + rg % w_leaders
            is_leader = 1 if my_leader == rg else 0
            if is_leader:
                for prg in range(first_leader):
                    if first_leader + prg % w_leaders == rg:
                        pc = device.worker_core_from_logical_core(cores[prg * G + g])
                        peers += [pc.x, pc.y]
        writer_rt[(core.x, core.y)] = [out_addr, head_start, num_heads, rg, w_addr, is_leader, len(peers) // 2] + peers
        compute_rt[core.x][core.y] = [num_heads, eps_bits]

    # Reader/writer NoCs: default (reader NOC_0, writer NOC_1) except the top NOC1_READ_ROWS logical rows, which
    # swap — same kernels, two core groups.
    dm_kernels = []
    # Carve-out: the swap relieves NoC0 starvation of the top rows, which needs many core rows below them issuing
    # reads. On a grid of few rows nothing is starved and the swap only moves reads onto the slower NoC1 path —
    # measured regressions at <= 6 used rows (128x512 +27%, 32x3072 w+rope +47%, 256x1024 w+rope ~+10%); measured
    # wins at 8 rows (256x3072) and 10 rows (2304x3072 w+rope -6%).
    used_rows = cores[-1].y + 1
    swap_rows = 0 if used_rows < NOC1_MIN_GRID_ROWS else NOC1_READ_ROWS
    swapped = [cc for cc in cores if cc.y < swap_rows]
    normal = [cc for cc in cores if cc.y >= swap_rows]
    for group, read_noc, write_noc in (
        (normal, ttnn.NOC.NOC_0, ttnn.NOC.NOC_1),
        (swapped, ttnn.NOC.NOC_1, ttnn.NOC.NOC_0),
    ):
        if not group:
            continue
        group_ranges = _core_ranges(group)
        group_reader_rt = ttnn.RuntimeArgs()
        group_writer_rt = ttnn.RuntimeArgs()
        for cc in group:
            group_reader_rt[cc.x][cc.y] = reader_rt[(cc.x, cc.y)]
            group_writer_rt[cc.x][cc.y] = writer_rt[(cc.x, cc.y)]
        dm_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_head_rope_reader.cpp"),
                core_ranges=group_ranges,
                compile_time_args=reader_ct,
                runtime_args=group_reader_rt,
                defines=defines,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=read_noc),
            )
        )
        dm_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_head_rope_writer.cpp"),
                core_ranges=group_ranges,
                compile_time_args=writer_ct,
                runtime_args=group_writer_rt,
                defines=defines,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=write_noc),
            )
        )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_head_rope_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        defines=defines,
        config=compute_config,
    )

    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_ROW_COUNTER, core_ranges=core_ranges, initial_value=Rg)]
    if weight_fwd:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_W_READY, core_ranges=core_ranges, initial_value=0))
    if team:
        semaphores += [
            ttnn.SemaphoreDescriptor(id=SEM_PEER_READY, core_ranges=core_ranges, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_ROW_MAILBOX, core_ranges=core_ranges, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_ROPE_VALID, core_ranges=core_ranges, initial_value=0),
        ]
    return ttnn.ProgramDescriptor(kernels=dm_kernels + [compute], semaphores=semaphores, cbs=cbs)
