# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for scaled_dot_product_attention (FlashAttention-2).

Blocking model (see op_design.md, binding):
  * Split B·H·q-chunks (all independent) across the grid — no cross-core combine.
  * Each core loops its assigned work units; per work unit it streams all KV
    chunks once and folds them into a running (m, l, O) online-softmax recurrence.
  * Every CB page count and kernel loop bound derives from three block knobs
    (Sq_chunk_t, Skv_chunk_t, KV_DEPTH) + Dt — no CB grows with S_q or S_kv.
"""

import os
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# ---- L1 budget (bytes) for the streaming working set. Conservative. ----
L1_BUDGET = 1_400_000

# ---- Compute-block factor targets (single source of truth; R3a) ----
# The block factor is the COARSEST chunk that fits L1 (design "coarsest that
# fits"). R3a raises the target from the phase-0 value 4 to 8 as the compute-side
# perf lever on the flagged compute-bound shape: a coarser chunk (a) amortizes the
# ~10 sequential per-chunk online-softmax helper phases (each pays a fixed
# reconfig/init/fill-drain) over 2x the tiles by halving n_kv_chunks, and (b) grows
# the QKᵀ matmul out_subblock_w toward the fp32_dest_acc_en=False DEST budget (8
# bf16 tiles) since out_subblock_w = decomp_n(Skv_chunk_t, dest_limit). `_fit_l1`
# shrinks from these targets under L1 pressure and `_chunk_size` caps per axis
# (and enforces the straddle-safe remainder), so raising the target is safe for
# every shape/dtype — the coarsest that fits is chosen, never larger than L1 or the
# axis tile-count allows.
SQ_CHUNK_TARGET = 8
SKV_CHUNK_TARGET = 8
KV_DEPTH_DEFAULT = 2
OUT_DEPTH_DEFAULT = 2

# ---- CB indices (semantic names; slots are just buffer indices) ----
CB_Q_IN = 0
CB_K_IN = 1
CB_V_IN = 2
CB_MASK_IN = 3
CB_SCALER = 4
CB_SCALE = 5
CB_M_NEW = 6  # scratch: new running-max before overwriting cb_row_max
CB_SUM_CHUNK = 7  # scratch: per-chunk row-sum before folding into l
CB_KV_MASK = 8  # last-KV-tile softmax padding mask (0 valid cols, -inf pad cols)
CB_OUT = 16
CB_Q_SCALED = 24
CB_SCORES = 25
CB_EXP = 26
CB_ROW_MAX = 27
CB_ROW_SUM = 28
CB_PV = 29
CB_OUT_ACCUM = 30
CB_CORR = 31


def _ceil_div(a, b):
    return (a + b - 1) // b


def _chunk_size(axis_t, target):
    """Largest chunk <= target that avoids an unsafe partial last chunk.

    Selection order:
      1. Largest EXACT DIVISOR of axis_t in (1, target]. A whole chunk on every
         step means NO partial last chunk, so the cb_scores / cb_exp ring never
         carries a fractional offset across work units. This is the common case for
         every real shape (tile-counts are almost always composite: powers of two,
         296 = 8·37, …) and is what the flagged perf shape rides (Skv_t=296 -> 8).
      2. Prime tile-count > target (e.g. Skv_t=101): no divisor in (1, target], so a
         partial last chunk is unavoidable if we want to beat the 1-tile collapse.
         Fall back to R1b's coarse chunk with a straddle-safe remainder (`rem | chunk`,
         i.e. `2*rem <= chunk`, which keeps the within-work-unit reduce/exp read
         window in-bounds): Skv_t=101 -> 5 (rem 1). This branch is reachable only by
         shapes with no golden cell AND a single work unit per core (verified), so the
         cross-work-unit ring carry below never triggers for them.

    Why divisors are PREFERRED (R3a): R1b used a coarse chunk + partial for every
    non-divisible tile-count and guarded only the *within*-work-unit ring straddle
    (`rem | chunk`). But a partial last chunk pushes `sq_valid*rem` tiles — a
    FRACTION of the `Sq_chunk_t*Skv_chunk_t`-page ring — so when a core runs >1 work
    unit (total_work > grid), the cb_scores/cb_exp read+write pointers start the next
    work unit mid-ring and the reduce's linear window straddles the wrap -> garbage
    (pcc≈0, rms=inf). R3a's coarser target (8) made the L1 shrink land previously-
    clean shapes (e.g. Skv_t=128, which 4 divides) on a partial chunk (6), exposing
    that latent cross-work-unit carry. Preferring a divisor removes partials for
    every shape that has one, so the ring realigns to slot 0 after each work unit.
    """
    hi = min(axis_t, target)
    # 1. largest exact divisor <= target (no partial chunk)
    for c in range(hi, 1, -1):
        if axis_t % c == 0:
            return c
    # 2. prime tile-count > target: R1b straddle-safe coarse + partial (avoids 1)
    for c in range(hi, 1, -1):
        rem = axis_t % c
        if c % rem == 0:  # rem != 0 here (no divisor found above)
            return c
    return 1


def _working_set_bytes(
    sq_chunk_t, skv_chunk_t, dt, kv_depth, out_depth, has_mask, gen_kv_mask, in_bytes, out_bytes, interm_bytes
):
    """Per-core L1 working set. R2: input/output tile bytes follow the tensor dtype
    (fp32 doubles vs bf16, bf8b halves) and the softmax intermediates
    (cb_scores/cb_exp) follow `interm_bytes` (fp32 under fp32-DEST accumulation)."""
    bf16 = ttnn.tile_size(ttnn.bfloat16)
    fp32 = ttnn.tile_size(ttnn.float32)
    total = 0
    total += sq_chunk_t * dt * in_bytes  # q_in
    total += skv_chunk_t * dt * kv_depth * in_bytes  # k_in
    total += skv_chunk_t * dt * kv_depth * in_bytes  # v_in
    if has_mask:
        total += sq_chunk_t * skv_chunk_t * kv_depth * in_bytes  # mask_in
    if gen_kv_mask:
        total += sq_chunk_t * skv_chunk_t * bf16  # cb_kv_mask (R1 KV-pad / R4 causal, one block)
    total += 2 * bf16  # scaler + scale (always bf16, reader-filled)
    total += sq_chunk_t * dt * bf16  # q_scaled (bf16 regardless of input dtype)
    total += sq_chunk_t * skv_chunk_t * interm_bytes  # scores
    total += sq_chunk_t * skv_chunk_t * interm_bytes  # exp
    total += sq_chunk_t * dt * out_depth * out_bytes  # out
    # Accumulators + sum_chunk follow interm_format (fp32 under fp32-DEST, bf16 in the
    # throughput regime) — see the CB block. In the 16-bit-DEST regime they are bf16.
    total += (sq_chunk_t * 5) * interm_bytes  # row_max, row_sum, corr, m_new, sum_chunk
    total += (sq_chunk_t * dt * 2) * interm_bytes  # pv, out_accum
    return total


def _fit_l1(sq_t, skv_t, dt, has_mask, gen_kv_mask, in_bytes, out_bytes, interm_bytes):
    """Compute the three block knobs once; shrink Skv_chunk_t -> KV_DEPTH -> Sq_chunk_t
    until the working set fits L1."""
    sq_target = SQ_CHUNK_TARGET
    skv_target = SKV_CHUNK_TARGET
    kv_depth = KV_DEPTH_DEFAULT
    out_depth = OUT_DEPTH_DEFAULT
    while True:
        sq_chunk_t = _chunk_size(sq_t, sq_target)
        skv_chunk_t = _chunk_size(skv_t, skv_target)
        need = _working_set_bytes(
            sq_chunk_t, skv_chunk_t, dt, kv_depth, out_depth, has_mask, gen_kv_mask, in_bytes, out_bytes, interm_bytes
        )
        if need <= L1_BUDGET:
            break
        if skv_target > 1:
            skv_target -= 1
        elif kv_depth > 1:
            kv_depth = 1
        elif sq_target > 1:
            sq_target -= 1
        else:
            break
    return sq_chunk_t, skv_chunk_t, kv_depth, out_depth


def _f32_bits(x):
    return struct.unpack("I", struct.pack("f", float(x)))[0]


def _cb(index, page_size, num_pages, data_format, core_grid):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


# ===========================================================================
# NoC-multicast (KV read-once + broadcast) variant — GUARDED, opt-out via env.
# ===========================================================================
# The roofline predicted this a no-win (reads ~93% hidden behind compute); this
# path was BUILT and MEASURED anyway (perf_findings.md § NoC-multicast). It is a
# SCHEME-CHANGE off the Blocking Model: Q-outer -> KV-outer, with each (b,h)
# grouped onto one ROW of the grid so ONE injector (column 0) reads each KV chunk
# once from DRAM and Mcast1D-broadcasts it across the row (10 groups x 11 cores on
# Blackhole = 110). Cuts DRAM K/V read volume ~cores_per_row-fold. Guarded to the
# shape class it is valid for; every other shape uses the shipped Q-outer path.
#
# MEASURED a +44% REGRESSION (perf_findings.md § NoC-multicast) — so OPT-IN only,
# never the default. Env knobs (measurement):
#   SDPA_MCAST=1            ENABLE the mcast path for eligible shapes (default: OFF,
#                           i.e. the fast Q-outer path handles every shape).
#   SDPA_MCAST_NO_BCAST=1   KV-outer WITHOUT broadcast (every core re-reads its own
#                           K/V from DRAM) — isolates the broadcast's effect (A/B).
#   SDPA_MCAST_ABLATE_READER=1 / SDPA_MCAST_ABLATE_WRITER=1
#                           NoC stubs for the KV-outer compute-floor probe.
MCAST_MAX_SUBCHUNK = 4  # resident q-chunk state sets the compute kernel allocates
MCAST_SQ_CHUNK_CAP = 8  # scores block height cap (keeps cb_scores/cb_exp small)
MCAST_SKV_CHUNK_CAP = 8
L1_BUDGET_MCAST = 1_450_000
MCAST_KV_DEPTH = 2  # double-buffer K/V so the injector prefetches chunk j+1
MCAST_OUT_DEPTH = 1  # outputs emitted once after the KV loop; no overlap needed

# mcast CB indices — must match the three mcast kernels.
MC_Q_STAGE = 0
MC_K_IN = 1
MC_V_IN = 2
MC_SCALER = 3
MC_SCALE = 4
MC_SCORES = 5
MC_EXP = 6
MC_CORR = 7
MC_M_NEW = 8
MC_SUM_CHUNK = 9
MC_Q_SCALED_0 = 10  # 10..13 (read-only, one per sub-chunk)
MC_PV = 14
MC_L_TMP = 15
MC_OUT = 16
MC_ROW_MAX = 17  # ROTATING: MAX_SUBCHUNK blocks (running m)
MC_ROW_SUM = 18  # ROTATING: MAX_SUBCHUNK blocks (running l)
MC_OUT_ACCUM = 19  # ROTATING: MAX_SUBCHUNK blocks (running O)
MC_O_TMP = 20
MC_M_OLD = 21  # depth-1 scratch (old max popped off the rotating ring)


def _mcast_pick_sq_chunk(sq_t, gx):
    """Largest Sq_chunk_t in [1, cap] dividing Sq_t s.t. n_q_chunks in
    [gx, gx*MAX_SUBCHUNK] (every row core active AND subchunk count <= cap)."""
    for c in range(min(MCAST_SQ_CHUNK_CAP, sq_t), 0, -1):
        if sq_t % c != 0:
            continue
        n_q = sq_t // c
        if gx <= n_q <= gx * MCAST_MAX_SUBCHUNK:
            return c, n_q
    return None, None


def _mcast_working_set(sq_chunk_t, skv_chunk_t, dt, in_bytes, out_bytes, interm_bytes):
    bf16 = ttnn.tile_size(ttnn.bfloat16)
    t = 0
    t += sq_chunk_t * dt * in_bytes  # q_stage (depth 1)
    t += skv_chunk_t * dt * MCAST_KV_DEPTH * in_bytes  # k_in
    t += skv_chunk_t * dt * MCAST_KV_DEPTH * in_bytes  # v_in
    t += 2 * bf16  # scaler + scale
    t += sq_chunk_t * skv_chunk_t * interm_bytes  # scores
    t += sq_chunk_t * skv_chunk_t * interm_bytes  # exp
    t += 3 * sq_chunk_t * interm_bytes  # corr, m_new, sum_chunk
    t += sq_chunk_t * dt * interm_bytes  # pv
    t += sq_chunk_t * dt * interm_bytes  # o_tmp
    t += sq_chunk_t * interm_bytes  # l_tmp
    t += sq_chunk_t * interm_bytes  # m_old
    t += sq_chunk_t * dt * MCAST_OUT_DEPTH * out_bytes  # out
    t += MCAST_MAX_SUBCHUNK * sq_chunk_t * dt * interm_bytes  # q_scaled (4 CBs)
    t += MCAST_MAX_SUBCHUNK * sq_chunk_t * dt * interm_bytes  # out_accum (rotating ring)
    t += MCAST_MAX_SUBCHUNK * sq_chunk_t * interm_bytes  # row_max (rotating ring)
    t += MCAST_MAX_SUBCHUNK * sq_chunk_t * interm_bytes  # row_sum (rotating ring)
    return t


def _mcast_pick_skv_chunk(skv_t, sq_chunk_t, dt, in_bytes, out_bytes, interm_bytes):
    """Largest Skv_chunk_t in [1, cap] dividing Skv_t whose working set fits L1."""
    for c in range(min(MCAST_SKV_CHUNK_CAP, skv_t), 0, -1):
        if skv_t % c != 0:
            continue
        if _mcast_working_set(sq_chunk_t, c, dt, in_bytes, out_bytes, interm_bytes) <= L1_BUDGET_MCAST:
            return c
    return None


def _mcast_eligible(query, key, value, attn_mask, is_causal, compute_kernel_config):
    """Return (params dict) if the mcast KV-outer path applies, else None.

    OPT-IN: the mcast variant was BUILT and MEASURED a +44% regression vs the
    shipped Q-outer path (perf_findings.md § NoC-multicast) — the KV-outer
    restructure raises the compute floor far more than hiding the (already-hidden)
    reads can recover. So it is NOT the default: the fast Q-outer path stays default
    for every shape, and the mcast path activates ONLY when SDPA_MCAST=1 AND the
    shape is in its valid class. This keeps the measured reference reproducible
    without regressing any shipped shape.
    """
    if os.environ.get("SDPA_MCAST") != "1":
        return None
    q = list(query.shape)
    k = list(key.shape)
    B, H, S_q, D = q
    H_kv, S_kv = k[1], k[-2]
    # Shape-class guard (the shape class the variant is valid for).
    if attn_mask is not None or is_causal:
        return None  # mask none only
    if query.dtype != ttnn.bfloat16 or key.dtype != ttnn.bfloat16:
        return None  # bf16 only (target dtype)
    if query.layout != ttnn.TILE_LAYOUT:
        return None
    if S_q != S_kv:
        return None  # self-attn only
    if H_kv != H:
        return None  # MHA only (each head reads distinct K/V)
    if S_q % TILE_DIM != 0 or D % TILE_DIM != 0:
        return None  # tile-aligned only
    fp32_dest = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))
    if fp32_dest:
        return None  # throughput (bf16-intermediate) regime only; fp32 interm won't fit

    device = query.device()
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    if gx < 2:
        return None
    num_groups = B * H
    if num_groups > gy:
        return None  # one (b,h) per row

    Sq_t = _ceil_div(S_q, TILE_DIM)
    Skv_t = _ceil_div(S_kv, TILE_DIM)
    Dt = _ceil_div(D, TILE_DIM)

    sq_chunk_t, n_q_chunks = _mcast_pick_sq_chunk(Sq_t, gx)
    if sq_chunk_t is None:
        return None

    in_bytes = ttnn.tile_size(query.dtype)
    out_bytes = ttnn.tile_size(query.dtype)
    interm_bytes = ttnn.tile_size(ttnn.bfloat16)
    skv_chunk_t = _mcast_pick_skv_chunk(Skv_t, sq_chunk_t, Dt, in_bytes, out_bytes, interm_bytes)
    if skv_chunk_t is None:
        return None

    return {
        "B": B,
        "H": H,
        "H_kv": H_kv,
        "Sq_t": Sq_t,
        "Skv_t": Skv_t,
        "Dt": Dt,
        "sq_chunk_t": sq_chunk_t,
        "skv_chunk_t": skv_chunk_t,
        "n_q_chunks": n_q_chunks,
        "n_kv_chunks": _ceil_div(Skv_t, skv_chunk_t),
        "gx": gx,
        "num_groups": num_groups,
    }


def _split_row(n_q_chunks, ncores):
    """Contiguous split of n_q_chunks across ncores -> [(q_start, q_cnt), ...]."""
    base = n_q_chunks // ncores
    rem = n_q_chunks % ncores
    out = []
    start = 0
    for i in range(ncores):
        cnt = base + (1 if i < rem else 0)
        out.append((start, cnt))
        start += cnt
    return out


def _create_mcast_program_descriptor(query, key, value, output_tensor, scale, compute_kernel_config, p):
    B, H, H_kv = p["B"], p["H"], p["H_kv"]
    Sq_t, Skv_t, Dt = p["Sq_t"], p["Skv_t"], p["Dt"]
    Sq_chunk_t, Skv_chunk_t = p["sq_chunk_t"], p["skv_chunk_t"]
    n_q_chunks, n_kv_chunks = p["n_q_chunks"], p["n_kv_chunks"]
    gx, num_groups = p["gx"], p["num_groups"]

    mcast_bcast = 0 if os.environ.get("SDPA_MCAST_NO_BCAST") == "1" else 1
    ablate_reader = 1 if os.environ.get("SDPA_MCAST_ABLATE_READER") == "1" else 0
    ablate_writer = 1 if os.environ.get("SDPA_MCAST_ABLATE_WRITER") == "1" else 0
    dest_limit = 8  # fp32_dest_acc_en=False throughput regime
    fast_exp = 1

    device = query.device()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, num_groups - 1))])
    sender_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_groups - 1))])
    recv_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(gx - 1, num_groups - 1))])

    # One Mcast1D row-family: sender in column 0, broadcasts across each row.
    mc = ttnn.Mcast1D(device, all_cores, ttnn.Mcast1DShape.PerRow, 0, ttnn.McastConfig(handshake=True, base_sem_id=0))
    semaphores = mc.owned_semaphores()

    bf16 = ttnn.tile_size(ttnn.bfloat16)
    in_page = query.buffer_page_size()
    out_page = output_tensor.buffer_page_size()
    interm = ttnn.bfloat16
    q_tiles = Sq_chunk_t * Dt
    kv_tiles = Skv_chunk_t * Dt
    score_tiles = Sq_chunk_t * Skv_chunk_t

    # ---- Circular buffers (on every participating core) ----
    # Rotating rings (row_max/row_sum/out_accum) hold MAX_SUBCHUNK blocks; the
    # compute kernel rotates them one block per sub-chunk. q_scaled stays as
    # MAX_SUBCHUNK separate CBs (read-only, read via In0SourceFn).
    cbs = [
        _cb(MC_Q_STAGE, in_page, q_tiles, query.dtype, all_cores),
        _cb(MC_K_IN, in_page, kv_tiles * MCAST_KV_DEPTH, query.dtype, all_cores),
        _cb(MC_V_IN, in_page, kv_tiles * MCAST_KV_DEPTH, query.dtype, all_cores),
        _cb(MC_SCALER, bf16, 1, ttnn.bfloat16, all_cores),
        _cb(MC_SCALE, bf16, 1, ttnn.bfloat16, all_cores),
        _cb(MC_SCORES, bf16, score_tiles, interm, all_cores),
        _cb(MC_EXP, bf16, score_tiles, interm, all_cores),
        _cb(MC_CORR, bf16, Sq_chunk_t, interm, all_cores),
        _cb(MC_M_NEW, bf16, Sq_chunk_t, interm, all_cores),
        _cb(MC_SUM_CHUNK, bf16, Sq_chunk_t, interm, all_cores),
        _cb(MC_PV, bf16, q_tiles, interm, all_cores),
        _cb(MC_L_TMP, bf16, Sq_chunk_t, interm, all_cores),
        _cb(MC_OUT, out_page, q_tiles * MCAST_OUT_DEPTH, output_tensor.dtype, all_cores),
        _cb(MC_ROW_MAX, bf16, Sq_chunk_t * MCAST_MAX_SUBCHUNK, interm, all_cores),
        _cb(MC_ROW_SUM, bf16, Sq_chunk_t * MCAST_MAX_SUBCHUNK, interm, all_cores),
        _cb(MC_OUT_ACCUM, bf16, q_tiles * MCAST_MAX_SUBCHUNK, interm, all_cores),
        _cb(MC_O_TMP, bf16, q_tiles, interm, all_cores),
        _cb(MC_M_OLD, bf16, Sq_chunk_t, interm, all_cores),
    ]
    for s in range(MCAST_MAX_SUBCHUNK):
        cbs.append(_cb(MC_Q_SCALED_0 + s, bf16, q_tiles, interm, all_cores))

    # ---- Reader CT (sender + receiver share the scalar prefix; IS_SENDER differs) ----
    def reader_ct(is_sender):
        ct = [
            Dt,
            Sq_chunk_t,
            Skv_chunk_t,
            n_kv_chunks,
            Sq_t,
            Skv_t,
            H,
            H_kv,
            _f32_bits(scale),
            1 if is_sender else 0,
            mcast_bcast,
            ablate_reader,
        ]
        ct += list(mc.compile_time_args())  # 5 words at index 12
        ct += ttnn.TensorAccessorArgs(query).get_compile_time_args()
        ct += ttnn.TensorAccessorArgs(key).get_compile_time_args()
        ct += ttnn.TensorAccessorArgs(value).get_compile_time_args()
        return ct

    # ---- Per-core runtime args ----
    sender_rt = ttnn.RuntimeArgs()
    recv_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()

    q_addr = query.buffer_address()
    k_addr = key.buffer_address()
    v_addr = value.buffer_address()
    o_addr = output_tensor.buffer_address()
    split = _split_row(n_q_chunks, gx)

    for y in range(num_groups):
        b = y // H
        h = y % H
        for x in range(gx):
            q_start, q_cnt = split[x]
            core = ttnn.CoreCoord(x, y)
            mc_rt = list(mc.runtime_args(core))
            base_rt = [q_addr, k_addr, v_addr, b, h, q_start, q_cnt]
            if x == 0:
                sender_rt[x][y] = base_rt + mc_rt
            else:
                recv_rt[x][y] = base_rt + mc_rt
            compute_rt[x][y] = [q_cnt]
            writer_rt[x][y] = [o_addr, b, h, q_start, q_cnt]

    kdir = str(KERNEL_DIR)
    sender_kernel = ttnn.KernelDescriptor(
        kernel_source=kdir + "/scaled_dot_product_attention_mcast_reader.cpp",
        core_ranges=sender_cores,
        compile_time_args=reader_ct(True),
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    receiver_kernel = ttnn.KernelDescriptor(
        kernel_source=kdir + "/scaled_dot_product_attention_mcast_reader.cpp",
        core_ranges=recv_cores,
        compile_time_args=reader_ct(False),
        runtime_args=recv_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_ct = [Dt, Sq_chunk_t, Skv_chunk_t, n_kv_chunks, MCAST_MAX_SUBCHUNK, dest_limit, fast_exp]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=kdir + "/scaled_dot_product_attention_mcast_compute.cpp",
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(compute_kernel_config, "math_fidelity", ttnn.MathFidelity.HiFi2),
            fp32_dest_acc_en=False,
            math_approx_mode=bool(getattr(compute_kernel_config, "math_approx_mode", False)),
            dst_full_sync_en=bool(getattr(compute_kernel_config, "dst_full_sync_en", False)),
        ),
    )
    writer_ct = [Dt, Sq_chunk_t, H, Sq_t, ablate_writer]
    writer_ct += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=kdir + "/scaled_dot_product_attention_mcast_writer.cpp",
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[sender_kernel, receiver_kernel, compute_kernel, writer_kernel],
        semaphores=list(semaphores),
        cbs=cbs,
    )
    return descriptor, [query, key, value]


def create_program_descriptor(
    query, key, value, output_tensor, *, attn_mask=None, is_causal=False, scale=1.0, compute_kernel_config=None
):
    # GUARDED NoC-multicast (KV read-once + broadcast) path. Only for the shape
    # class it is valid for; every other shape falls through to the shipped
    # Q-outer path below unchanged. Opt-out via SDPA_MCAST=0.
    mcast_params = _mcast_eligible(query, key, value, attn_mask, is_causal, compute_kernel_config)
    if mcast_params is not None:
        return _create_mcast_program_descriptor(
            query, key, value, output_tensor, scale, compute_kernel_config, mcast_params
        )

    q_shape = list(query.shape)
    k_shape = list(key.shape)

    B, H, S_q, D = q_shape
    H_kv = k_shape[1]
    S_kv = k_shape[-2]

    Sq_t = _ceil_div(S_q, TILE_DIM)
    Skv_t = _ceil_div(S_kv, TILE_DIM)
    Dt = _ceil_div(D, TILE_DIM)

    has_mask = attn_mask is not None
    mask_H = attn_mask.shape[1] if has_mask else 0

    # R1 (h_non_aligned): valid columns in the last S_kv tile (0 => aligned).
    # When non-zero, the last KV chunk's boundary tile carries an additive -inf
    # mask on its padding columns so they fall out of the softmax reduce.
    skv_partial = S_kv % TILE_DIM
    has_kv_pad = skv_partial != 0

    # R4 (causal): the triangular −∞ bias is generated on-device (no mask tensor).
    # It reuses the same generated-mask CB (cb_kv_mask) + additive compute path as
    # R1's KV-padding mask. For causal self-attention the diagonal mask already
    # drives every padding key (index >= S_kv) to −∞ (a padding key is always in the
    # future of every valid query), so causal SUBSUMES the KV-padding mask — the
    # kernels prefer the causal path when is_causal and never double-generate.
    causal = bool(is_causal)
    gen_kv_mask = has_kv_pad or causal

    # R2: precision surface. Input/output CB tile bytes follow the tensor dtype;
    # the softmax intermediates (cb_scores/cb_exp) are fp32 under fp32-DEST
    # accumulation (the precision lever that pushes adversarial distributions back
    # under tolerance — verification_report.md), else bf16 so the perf-flagged
    # bf16 + 16-bit-DEST regime stays byte-identical. Never bf8b: block-float
    # intermediates would compound quantization through the online softmax.
    fp32_dest = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))
    in_bytes = ttnn.tile_size(query.dtype)
    out_bytes = ttnn.tile_size(output_tensor.dtype)
    interm_format = ttnn.float32 if fp32_dest else ttnn.bfloat16
    interm_bytes = ttnn.tile_size(interm_format)

    Sq_chunk_t, Skv_chunk_t, KV_DEPTH, OUT_DEPTH = _fit_l1(
        Sq_t, Skv_t, Dt, has_mask, gen_kv_mask, in_bytes, out_bytes, interm_bytes
    )

    n_q_chunks = _ceil_div(Sq_t, Sq_chunk_t)
    n_kv_chunks = _ceil_div(Skv_t, Skv_chunk_t)
    total_work = B * H * n_q_chunks

    # R3e (perf): fuse the per-chunk row-sum reduce into the exp pack (raw-LLK dual-pack). Gated to
    # the fp32_dest_acc_en=False throughput regime (bf16 softmax intermediates, so the dual-pack
    # targets share a format); the max-precision fp32-DEST regime keeps the exact per-chunk reduce.
    fuse_rowsum = 0 if fp32_dest else 1
    # Perf A/B knob (measurement only, defaults to no-op): SDPA_FUSE_ROWSUM=0 forces the pre-R3e
    # reduce<SUM> row-sum path even in the throughput regime so the fused dual-pack can be compared
    # same-session against its own baseline (defeats AICLK drift between fresh invocations).
    if os.environ.get("SDPA_FUSE_ROWSUM") == "0":
        fuse_rowsum = 0
    # R6 (perf): fuse the online-softmax O-accumulate (former compute phase 10) into the PV matmul
    # via packer L1-accumulation onto cb_out_accum. Gated to the throughput regime (fuse_rowsum)
    # AND no-partial-q-chunk (Sq_t % Sq_chunk_t == 0): only then is cb_out_accum a FULL ring, so
    # phase 8's in-place rescale wraps the packer write pointer back onto the resident alpha*O for
    # the matmul to L1-accumulate P*V onto in place. When fused, cb_pv is not needed (the PV matmul
    # packs straight onto cb_out_accum) and phase 10 disappears — see the compute kernel. The
    # fp32-DEST path and the rare prime-Sq_t partial-q throughput path keep cb_pv + phase 10.
    fuse_oaccum = (fuse_rowsum == 1) and ((Sq_t % Sq_chunk_t) == 0)

    # DEST tile budget: fp32 accumulation halves the 8-tile bf16 budget. The matmul
    # N-subblock decomposition (out_subblock_w <= dest_limit) is derived on-device
    # (R1b) so the partial last chunk's runtime N re-derives cleanly — single source
    # of truth in the compute kernel's decomp_n().
    dest_limit = 4 if fp32_dest else 8

    # ---- Grid + work distribution ----
    device = query.device()
    grid = device.compute_with_storage_grid_size()
    num_cores, all_cores, group1, group2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, total_work, True)

    g1_cores = ttnn.corerange_to_cores(group1, None, True) if group1.num_cores() > 0 else []
    g2_cores = ttnn.corerange_to_cores(group2, None, True) if group2.num_cores() > 0 else []
    ordered = list(g1_cores) + list(g2_cores)
    counts = [per_core_1] * len(g1_cores) + [per_core_2] * len(g2_cores)

    bf16 = ttnn.tile_size(ttnn.bfloat16)
    fp32 = ttnn.tile_size(ttnn.float32)
    in_page = query.buffer_page_size()
    out_page = output_tensor.buffer_page_size()

    # ---- Circular buffers ----
    cbs = [
        _cb(CB_Q_IN, in_page, Sq_chunk_t * Dt, query.dtype, all_cores),
        _cb(CB_K_IN, in_page, Skv_chunk_t * Dt * KV_DEPTH, query.dtype, all_cores),
        _cb(CB_V_IN, in_page, Skv_chunk_t * Dt * KV_DEPTH, query.dtype, all_cores),
        _cb(CB_SCALER, bf16, 1, ttnn.bfloat16, all_cores),
        _cb(CB_SCALE, bf16, 1, ttnn.bfloat16, all_cores),
        # Accumulators follow interm_format (fp32 under fp32-DEST, bf16 in the throughput
        # regime). In fp32_dest_acc_en=False the DEST is 16-bit, so every op's arithmetic is
        # bf16-precision regardless — an fp32 L1 accumulator only holds a bf16 value in an
        # fp32 container and forces a bf16<->fp32 reconfig at each phase boundary for no
        # precision gain. Matching interm_format makes the throughput chain uniform-bf16
        # (numerically identical: the values were already bf16-bound by the DEST), eliding
        # those reconfigs and halving the accumulators' L1. fp32-DEST regime: interm_format
        # is fp32 -> unchanged.
        _cb(CB_M_NEW, interm_bytes, Sq_chunk_t, interm_format, all_cores),
        # R3e: cb_sum_chunk carries the per-chunk partial row-sum. In the fused
        # (fp32_dest_acc_en=False) regime it is the L1-accumulation target of the
        # exp dual-pack, so it MUST share cb_exp's data format (interm_format) — the
        # two packs then need no pack_reconfig_data_format between them. In the
        # non-fused (fp32-DEST) regime interm_format is fp32, so this is unchanged.
        _cb(CB_SUM_CHUNK, interm_bytes, Sq_chunk_t, interm_format, all_cores),
        _cb(CB_OUT, out_page, Sq_chunk_t * Dt * OUT_DEPTH, output_tensor.dtype, all_cores),
        _cb(CB_Q_SCALED, bf16, Sq_chunk_t * Dt, ttnn.bfloat16, all_cores),
        # cb_scores / cb_exp: fp32 under fp32-DEST accumulation (softmax precision),
        # bf16 otherwise. Consumed by FPU ops (add/reduce/sub/matmul) so they are
        # NOT UnpackToDestFp32-tagged — the L1 format alone lifts the intermediate
        # from bf16 (7-bit) to fp32 (unpacks to TF32, 10-bit) through the pipeline.
        _cb(CB_SCORES, interm_bytes, Sq_chunk_t * Skv_chunk_t, interm_format, all_cores),
        _cb(CB_EXP, interm_bytes, Sq_chunk_t * Skv_chunk_t, interm_format, all_cores),
        _cb(CB_ROW_MAX, interm_bytes, Sq_chunk_t, interm_format, all_cores),
        _cb(CB_ROW_SUM, interm_bytes, Sq_chunk_t, interm_format, all_cores),
        _cb(CB_OUT_ACCUM, interm_bytes, Sq_chunk_t * Dt, interm_format, all_cores),
        _cb(CB_CORR, interm_bytes, Sq_chunk_t, interm_format, all_cores),
    ]
    if not fuse_oaccum:
        # R6: cb_pv holds the PV result before the phase-10 accumulate. Not needed in the fused
        # regime (the PV matmul L1-accumulates straight onto cb_out_accum). Allocated for the
        # fp32-DEST + partial-q throughput paths that keep phase 10. NOTE: _working_set_bytes still
        # counts cb_pv unconditionally (conservative) so the block-knob selection stays byte-
        # identical to the pre-R6 tree — the fused regime just frees the block at runtime.
        cbs.append(_cb(CB_PV, interm_bytes, Sq_chunk_t * Dt, interm_format, all_cores))
    if has_mask:
        cbs.append(_cb(CB_MASK_IN, in_page, Sq_chunk_t * Skv_chunk_t * KV_DEPTH, query.dtype, all_cores))
    if gen_kv_mask:
        # One score-block worth of generated additive-mask tiles (bf16). R1 fills the
        # last S_kv column tiles with the −∞ padding mask; R4 fills the straddling
        # KV chunk with the triangular causal mask. Either way it is generated by the
        # reader once per work unit on the relevant chunk(s) and consumed in place by
        # the additive-mask compute phase.
        cbs.append(_cb(CB_KV_MASK, bf16, Sq_chunk_t * Skv_chunk_t, ttnn.bfloat16, all_cores))

    # ---- Reader kernel ----
    # Perf (MEASUREMENT-ONLY, /perf-measure classify-the-bound): reader NoC stub. Skips
    # every noc_async_read_tile + barrier in read_tiles while keeping cb_reserve/push
    # intact, so DRAM bytes moved -> 0 but the CB producer/consumer counts (and compute)
    # are unchanged. A flat wall-time with reads stubbed proves the reads are hidden behind
    # compute (compute-bound); a large drop would prove DM-bound. Unset => 0 (shipped path,
    # byte-identical). SDPA_ABLATE_READER=1 stubs the reads (same family as SDPA_ABLATE_PV).
    ablate_reader = 1 if os.environ.get("SDPA_ABLATE_READER") == "1" else 0
    reader_ct = [
        B,
        H,
        H_kv,
        Sq_t,
        Skv_t,
        Dt,
        Sq_chunk_t,
        Skv_chunk_t,
        n_q_chunks,
        n_kv_chunks,
        mask_H,
        1 if has_mask else 0,
        _f32_bits(scale),
        skv_partial,
        1 if causal else 0,
        ablate_reader,
    ]
    reader_ct += ttnn.TensorAccessorArgs(query).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(key).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(value).get_compile_time_args()
    reader_ct += (
        ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    start = 0
    for core, cnt in zip(ordered, counts):
        reader_rt[core.x][core.y] = [
            query.buffer_address(),
            key.buffer_address(),
            value.buffer_address(),
            attn_mask.buffer_address() if has_mask else 0,
            start,
            cnt,
        ]
        writer_rt[core.x][core.y] = [output_tensor.buffer_address(), start, cnt]
        # start_wu lets compute decode qc per work unit -> sq_valid (partial q-chunk).
        compute_rt[core.x][core.y] = [cnt, start]
        start += cnt

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer kernel ----
    # Perf (MEASUREMENT-ONLY, writer twin of SDPA_ABLATE_READER): output NoC stub. Skips
    # every noc_async_write_tile + barrier in write_tiles while keeping cb_wait_front/pop
    # intact, so output DRAM bytes moved -> 0 but the compute->writer CB counts are
    # unchanged. A flat wall-time with writes stubbed proves the output writes are hidden
    # behind compute. Unset => 0 (shipped path, byte-identical). SDPA_ABLATE_WRITER=1 stubs.
    ablate_writer = 1 if os.environ.get("SDPA_ABLATE_WRITER") == "1" else 0
    writer_ct = [B, H, Sq_t, Dt, Sq_chunk_t, n_q_chunks, ablate_writer]
    writer_ct += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Compute kernel ----
    # R1b: matmul subblocks are derived on-device from the per-chunk runtime tile
    # counts (sq_valid = M, skv_valid = QKᵀ N / PV K), so only the block knobs +
    # axis tile-counts + dest_limit are threaded (no host subblock CT args).
    # R3c (perf): use the fast/approximate SFPU exp for the dominant softmax P=exp
    # phase ONLY in the 16-bit-DEST throughput regime (fp32_dest_acc_en=False). Zone
    # profiling showed the exact exp is ~54% of per-chunk compute; the fast exp is
    # ~75% cheaper (flagged shape 9.01->5.80 ms = 1.55x, PCC 0.997 held). The fast
    # exp adds ~0.003-0.02 normalized-RMS, which the fp32_dest_acc_en=False golden
    # tolerances (0.12 for bf16/bf8b) absorb, but the max-precision fp32_dest_acc_en=
    # True tolerances (fp32 0.02, bf16 0.05) do NOT — so those stay EXACT (byte-
    # identical, no regression). The flagged perf shape is fp32_dest_acc_en=False,
    # so it gets the full speedup. The alpha-correction exp (phase 5) stays exact
    # regardless (protects the online-softmax running (m, l, O) across chunks).
    fast_exp = 0 if fp32_dest else 1
    # R3e (perf): fuse_rowsum (the raw-LLK exp+row-sum dual-pack) is computed above with the CB
    # list (cb_pv allocation keys on the derived fuse_oaccum). Same gate as fast_exp.
    # R5 (perf): PV matmul output-subblock HEIGHT knob — grow out_subblock_h to fill the DEST
    # budget (the compute kernel's decomp_h: h = dest_limit/out_subblock_w when the output is
    # single-N-subblock, so the PV matmul (N=Dt=4, dest_limit=8 in the fp32_dest_acc_en=False
    # throughput regime) would use the full 8-tile DEST per pass instead of 4, halving the
    # subblock/pack passes). MEASURED same-session A/B on the flagged 1x10x9472x128 shape:
    # h=2 (on) 5.461 ms vs h=1 (off) 5.443 ms — FLAT/marginally negative. Root cause: filling
    # the full 8-tile half-sync DEST section per subblock defeats the intra-DEST math/pack
    # pipeline that h=1 (4-tile subblocks, 4 tiles free) enables. So this correct, general,
    # self-gating lever is PARKED at its trivial default (grow_subblock_h=0 => h=1, byte-
    # identical to R4) — the decomp_h scaffolding stays a live knob (SDPA_PV_SB_H=1 re-enables
    # it same-session for re-measurement, e.g. under a future full-sync-DEST or FPU∥SFPU
    # overlap scheme that would expose the pack overhead). Unset => parked (off).
    grow_subblock_h = 0
    if os.environ.get("SDPA_PV_SB_H") == "1":
        grow_subblock_h = 1
    # R5a (perf, MEASUREMENT-ONLY): PV-matmul + O-rescale/accumulate ablation gate. Bounds
    # the headroom any PV-batching lever could remove by stubbing that payload while keeping
    # CB sync intact (/perf-measure). SDPA_ABLATE_PV=1 stubs the PV matmul; =2 also stubs the
    # per-chunk O rescale + accumulate. Unset/other => 0 (shipped path, byte-identical).
    ablate_pv = 0
    _abl = os.environ.get("SDPA_ABLATE_PV")
    if _abl in ("1", "2", "3"):
        ablate_pv = int(_abl)
    # Perf 2 (MEASUREMENT-ONLY): stub the softmax payloads (row-max reduce + exp dual-pack)
    # keeping CB sync intact. Combined with SDPA_ABLATE_PV=3 this measures the pure per-phase
    # overhead floor vs the softmax payload. =2 ALSO stubs the online-recurrence phases P05
    # (max-update + alpha) and P07 (row-sum update) so the ENTIRE per-chunk KV loop is empty
    # CB bookkeeping (no wait-sink phase) — a "fully stubbed" per-chunk pipeline. Only valid in
    # the throughput regime (fuse_rowsum=True). Unset/other => 0 (shipped path, byte-identical).
    _abs = os.environ.get("SDPA_ABLATE_SOFTMAX")
    ablate_softmax = int(_abs) if _abs in ("1", "2") else 0
    # Perf (MEASUREMENT-ONLY): per-phase profiling zones. When SDPA_ZONE_PROFILE=1, inject
    # -DSDPA_ZONE_PROFILE so the compute kernel's DeviceZoneScopedN per-phase markers compile
    # in (recording begin/end cycles per phase per RISC into profile_log_device.csv). Absent by
    # default -> the zone macros are no-ops -> shipped build byte-identical (never perturbs the
    # perf harness). Attributes the compute-bound residual across the serialized helper phases.
    compute_defines = [("SDPA_ZONE_PROFILE", "1")] if os.environ.get("SDPA_ZONE_PROFILE") == "1" else []
    compute_ct = [
        Dt,
        Sq_chunk_t,
        Skv_chunk_t,
        n_kv_chunks,
        1 if has_mask else 0,
        skv_partial,
        Sq_t,
        n_q_chunks,
        Skv_t,
        dest_limit,
        fast_exp,
        fuse_rowsum,
        1 if causal else 0,
        grow_subblock_h,
        ablate_pv,
        ablate_softmax,
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        defines=compute_defines,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            # R2: full compute-config surface threaded from the caller's config. The
            # defaults reproduce default_compute_kernel_config() (HiFi4 + fp32 DEST +
            # no approx + half-sync) so callers passing nothing see identical results.
            math_fidelity=getattr(compute_kernel_config, "math_fidelity", ttnn.MathFidelity.HiFi4),
            fp32_dest_acc_en=fp32_dest,
            math_approx_mode=bool(getattr(compute_kernel_config, "math_approx_mode", False)),
            dst_full_sync_en=bool(getattr(compute_kernel_config, "dst_full_sync_en", False)),
        ),
    )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
    ordered_inputs = [query, key, value] + ([attn_mask] if has_mask else [])
    return descriptor, ordered_inputs
