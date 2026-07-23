# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for scaled_dot_product_attention (Flash Attention).

Work unit = one Q-block = (batch, query-head, Q-chunk). The flat work-list
``total_q_blocks = B · H_q · q_num_chunks`` is split across the grid with
``ttnn.split_work_to_cores``; each core streams every KV-block for each Q-block
it owns (online softmax). No cross-core communication in Phase-1.

Block-size knobs (single source of truth):
  * ``Q_CHUNK_TILES`` / ``K_CHUNK_TILES`` — per-core Q/KV chunk sizes (tiles).
    The effective chunk is the largest divisor of Sqt / Skvt that is ≤ the knob
    (keeps every chunk full — no partial-tail chunk in Phase-1). CB page counts
    and loop bounds are all derived from these.
  * ``KV_BUFFER_FACTOR`` / ``Q_BUFFER_FACTOR`` — streaming CB depths.
"""

import os
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Block-size / buffer-depth knobs (parameters, never inlined downstream) ----
# Base block factors (Refinements 0–2): the grid-fill-neutral pair. `_pick_chunks`
# selects the coarsest divisor pair <= these that fits L1 (D<=256 keep (4,4); large
# head_dim shrinks to fit). This is the byte-identical prior-phase behavior.
Q_CHUNK_TILES = 4  # Sq_chunk_t base upper bound (tiles of 32 query rows)
K_CHUNK_TILES = 4  # Sk_chunk_t base upper bound (tiles of 32 key/value rows)

# Refinement 3b (perf, compute-side amortization): the online-softmax kv_step runs
# ~7 sequential helper phases per KV-block, each paying an init/dst-sync/format-
# reconfig tax; on the flagged profile (~74 KV-blocks × ~7 Q-blocks/core) that fixed
# per-block cost dominates (the shape is compute/dataflow-latency bound, not read-
# bound — Refinement 3 proved this on device). COARSENING the chunk pair amortizes the
# fixed per-phase overhead over more tiles per call (master.md `compute_block_size` —
# the win grows with the phase count, which is exactly SDPA's many-phase kv_step).
# MEASURED on the flagged shape (1,10,9472,128) bf16 @ fp32_dest_acc_en=False: the win
# is NON-MONOTONIC — only the (8,8) PAIR beats the (4,4) baseline (11.04→10.24 ms,
# 1.078×); coarsening one axis alone is SLOWER ((4,8)=12.31 ms, (8,4)=12.10 ms). So
# 3b is a binary regime switch to the coarse pair, taken ONLY when it (a) is a real
# divisor pair, (b) fits L1, and (c) keeps the grid filled (q-coarsening shrinks the
# flat Q-block work-list). Any shape that doesn't qualify stays byte-identical to the
# Refinement-2 baseline pick — no correctness OR grid-fill regression by construction.
Q_CHUNK_COARSE = 8  # Sq_chunk_t coarse (amortization) block factor
K_CHUNK_COARSE = 8  # Sk_chunk_t coarse (amortization) block factor
# kv_buffer_factor: double-buffer K/V/mask (overlap DRAM read with compute). Refinement
# 3b measured depth 3 on the flagged shape at 10.29 ms vs 10.24 ms at depth 2 — no gain
# (slightly worse from L1 pressure): reads are OFF the critical path here (R3 proved the
# shape is compute/dataflow-latency bound, not read-bound), so a deeper read buffer is a
# no-op. Kept at the double-buffer default (a live tunable for any future read-bound shape).
KV_BUFFER_FACTOR = 2
Q_BUFFER_FACTOR = 1  # Q held resident across the whole KV loop

# ---- Per-core L1 budget (Refinement 2) ----------------------------------------
# The flash-attention CBs (Q/K/V/out + running-(m,l,O) accumulators) scale with
# `sq_chunk_t·dht` / `sk_chunk_t·dht`; D is never split across cores, so as head_dim
# grows the per-core footprint eventually exceeds L1. `sq_chunk_t`/`sk_chunk_t` are
# the block-factor knobs the design's Blocking Model exposes — cap them to the
# COARSEST divisor pair whose CB footprint fits the device L1 CB arena.
#
# The allocator lays CBs out as   grow_to = L1_CB_RESERVED + Σ(CB total_size)
# and rejects the program when grow_to > L1_CB_CEILING. Both constants below were
# measured exact-to-the-byte on device (Wormhole; see probes/probe_005/006) across
# bf16/fp32/bf8b × {mask,no-mask}: grow_to always equals RESERVED + our own CB-size
# sum. We keep a safety margin for any per-shape variation in the reserved region.
L1_CB_CEILING = 1572864  # device max CB top address (bytes)
L1_CB_RESERVED = 111360  # constant region below the CB arena (kernel bins / RT args)
L1_CB_SAFETY = 32 * 1024  # headroom for reserved-region variation across configs
L1_CB_BUDGET = L1_CB_CEILING - L1_CB_RESERVED - L1_CB_SAFETY

# ---- CB indices (semantic) ----
CB_Q_IN = 0
CB_K_IN = 1
CB_V_IN = 2
CB_MASK_IN = 3
CB_SCALER_MAX = 4
CB_SCALER_SUM = 5
CB_OUT = 16
# intermediates
CB_QK_SCORES = 24
CB_MAX_A = 25
CB_MAX_B = 26
CB_MAX_NEW = 27
CB_SUM_A = 28
CB_SUM_B = 29
CB_SUM_NEW = 30
CB_EXP_MAX_DIFF = 31
CB_OUT_A = 6
CB_OUT_B = 7
CB_OUT_NEW = 8
CB_SUM_SCALED = 9
CB_OUT_SCALED = 10


def _divisors_leq(n, cap):
    """All divisors d of n with 1 <= d <= cap, ascending (keeps every chunk full —
    no partial-tail chunk in Phase-1)."""
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def _cb_specs(sq, sk, dht, needs_mask, in_df, out_df, interm_df, scaler_df, mask_df):
    """(index, num_pages, data_format) for every CB — the SINGLE source of truth
    shared by the L1-footprint budget calc and the actual CB build below. Page
    counts are functions of the block factors sq/sk and DHt (never the full S).

    ``needs_mask`` allocates cb_mask_in for BOTH the custom (streamed DRAM mask)
    and causal (on-device generated mask) regimes; ``mask_df`` is that CB's
    format (in_df for custom — the reader byte-copies DRAM mask tiles; interm_df
    for causal — the reader generates −inf/0 tiles matching the score format)."""
    q_tiles = sq * dht
    k_tiles = sk * dht
    qk_tiles = sq * sk
    o_tiles = sq * dht
    specs = [
        (CB_Q_IN, q_tiles * Q_BUFFER_FACTOR, in_df),
        (CB_K_IN, k_tiles * KV_BUFFER_FACTOR, in_df),
        (CB_V_IN, k_tiles * KV_BUFFER_FACTOR, in_df),
        (CB_SCALER_MAX, 1, scaler_df),
        (CB_SCALER_SUM, 1, scaler_df),
        (CB_OUT, o_tiles * 2, out_df),
        (CB_QK_SCORES, qk_tiles, interm_df),
        (CB_MAX_A, sq, interm_df),  # running max (cb_m)
        (CB_MAX_B, sq, interm_df),  # m_cur scratch (cb_max_cur)
        (CB_MAX_NEW, sq, interm_df),
        (CB_SUM_A, sq, interm_df),  # running sum (cb_l)
        (CB_SUM_NEW, sq, interm_df),
        (CB_SUM_SCALED, sq, interm_df),
        (CB_EXP_MAX_DIFF, sq, interm_df),
        (CB_OUT_A, o_tiles, interm_df),  # running output accumulator (cb_o)
        (CB_OUT_NEW, o_tiles, interm_df),
        (CB_OUT_SCALED, o_tiles, interm_df),
    ]
    if needs_mask:
        specs.append((CB_MASK_IN, qk_tiles * KV_BUFFER_FACTOR, mask_df))
    return specs


def _cb_footprint_bytes(specs):
    return sum(num_pages * ttnn.tile_size(df) for _, num_pages, df in specs)


def _fits_l1(sq, sk, dht, needs_mask, in_df, out_df, interm_df, scaler_df, mask_df):
    return (
        _cb_footprint_bytes(_cb_specs(sq, sk, dht, needs_mask, in_df, out_df, interm_df, scaler_df, mask_df))
        <= L1_CB_BUDGET
    )


def _pick_chunks(sqt, skvt, dht, needs_mask, in_df, out_df, interm_df, scaler_df, mask_df, bh=1, num_cores=1):
    """Block-factor pair (sq_chunk_t, sk_chunk_t) for this shape.

    Refinement 3b — compute-side amortization: FIRST try the coarse pair
    (Q_CHUNK_COARSE, K_CHUNK_COARSE). The measured win on the flagged profile is
    NON-MONOTONIC — only the full coarse PAIR beats the baseline; coarsening one
    axis alone is slower — so this is a binary regime switch, taken only when the
    coarse pair (a) is a real divisor pair of (sqt, skvt), (b) fits the L1 budget,
    and (c) keeps the flat Q-block work-list filling the grid after q-coarsening
    (`bh · sqt/Q_CHUNK_COARSE >= num_cores`). If any condition fails the shape falls
    through to the Refinement-2 baseline below — byte-identical to prior phases, so
    no correctness or grid-fill regression is possible for a non-qualifying shape.

    Baseline (Refinements 0–2): the coarsest (largest-footprint) divisor pair
    <= (Q_CHUNK_TILES, K_CHUNK_TILES) whose per-core CB footprint fits L1 (D<=256
    keep (4,4); large head_dim shrinks to fit). Falls back to (1,1) if even that OOMs.
    """
    fits = lambda sq, sk: _fits_l1(sq, sk, dht, needs_mask, in_df, out_df, interm_df, scaler_df, mask_df)

    cq, ck = Q_CHUNK_COARSE, K_CHUNK_COARSE
    if sqt % cq == 0 and skvt % ck == 0 and fits(cq, ck) and bh * (sqt // cq) >= num_cores:
        return cq, ck

    best = None  # (footprint, sq, sk)
    for sq in _divisors_leq(sqt, Q_CHUNK_TILES):
        for sk in _divisors_leq(skvt, K_CHUNK_TILES):
            fp = _cb_footprint_bytes(_cb_specs(sq, sk, dht, needs_mask, in_df, out_df, interm_df, scaler_df, mask_df))
            if fp <= L1_CB_BUDGET and (best is None or fp > best[0]):
                best = (fp, sq, sk)
    if best is None:
        return 1, 1
    return best[1], best[2]


def _pick_subblock(m_tiles, n_tiles, dst_limit):
    """Pick (sb_h, sb_w) with sb_h*sb_w <= dst_limit, sb_w | n, sb_h | m.
    Returns (sb_h, sb_w, in0_num_subblocks=m/sb_h, in1_num_subblocks=n/sb_w)."""
    best = (1, 1)
    for w in range(min(n_tiles, dst_limit), 0, -1):
        if n_tiles % w != 0:
            continue
        max_h = dst_limit // w
        for h in range(min(m_tiles, max_h), 0, -1):
            if m_tiles % h == 0:
                best = (h, w)
                return best[0], best[1], m_tiles // best[0], n_tiles // best[1]
    h, w = best
    return h, w, m_tiles // h, n_tiles // w


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _resolve_math_fidelity(dtype, requested):
    """Correct fidelity per input dtype (single source of truth).

    bf16 / bf8b carry a 7-bit mantissa that fits losslessly in the FPU's TF32
    src registers, so HiFi3/HiFi4's extra passes buy nothing — and HiFi4 +
    fp32-DEST with bf16 inputs *silently corrupts* the matmul (issue #38306).
    Clamp those inputs to HiFi2. float32 keeps the requested fidelity (HiFi4
    recovers the mantissa bits TF32 truncation drops).
    """
    if dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        if requested in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi3):
            return ttnn.MathFidelity.HiFi2
    return requested


def create_program_descriptor(
    query, key, value, attn_mask, output_tensor, *, scale, is_causal=False, compute_kernel_config
):
    b, h_q, s_q, d = tuple(query.shape)
    _, h_kv, s_kv, _ = tuple(key.shape)

    sqt = s_q // 32
    skvt = s_kv // 32
    dht = d // 32

    # ---- Mask regime (Refinement 4) ----
    # 0=none, 1=custom (stream the caller's additive mask from DRAM), 2=causal
    # (generate the triangular −inf bias on-device + truncate the KV loop). custom
    # and causal are mutually exclusive (validate() enforces is_causal ⊕ attn_mask).
    has_mask = attn_mask is not None  # custom regime (streamed DRAM mask tensor)
    is_causal = bool(is_causal)
    mask_regime = 2 if is_causal else (1 if has_mask else 0)
    needs_mask_cb = has_mask or is_causal  # both regimes allocate + consume cb_mask_in
    mask_broadcast_head = 1 if (has_mask and tuple(attn_mask.shape)[1] == 1) else 0

    fp32_dest_acc_en = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))
    dst_limit = 4 if fp32_dest_acc_en else 8

    # ---- Circular-buffer formats (dtype-derived; single source per role) ----
    # (resolved here — before the block-factor pick — because the CB footprint,
    #  hence the L1-budget cap on sq_chunk_t/sk_chunk_t, depends on the tile bytes
    #  of the resolved dtype; float32's 2x tile size lowers the D at which OOM
    #  strikes, so the budget must see the true tile size. See the CB build below
    #  for the per-role rationale.)
    in_df = query.dtype
    out_df = output_tensor.dtype  # follows the input dtype (see entry point)
    interm_df = ttnn.float32 if in_df == ttnn.float32 else ttnn.bfloat16
    scaler_df = ttnn.bfloat16
    # cb_mask_in format: custom copies DRAM mask tiles (input dtype); causal
    # generates −inf/0 tiles that must match the score block's format (interm_df)
    # so the compute-side `add` sees identical formats (no reconfig surprise).
    mask_df = interm_df if is_causal else in_df

    # Grid is needed before the block-factor pick (the grid-fill guard on sq keeps
    # the flat Q-block work-list from underfilling the grid when sq is coarsened).
    device = query.device()
    grid = device.compute_with_storage_grid_size()
    grid_cols, grid_rows = grid.x, grid.y
    num_cores = grid_cols * grid_rows

    # ---- L1-budget cap on the block factors (Refinement 2) + coarsening (3b) ----
    # sq_chunk_t/sk_chunk_t are the design's block-factor knobs; pick the coarsest
    # divisor pair whose CB footprint fits L1 AND (for sq) keeps the grid filled.
    # Knob-turn only — the kernel is already parameterized on (sq, sk, dht); shrinking
    # the chunk just adds more (fully-full) Q/KV chunks to the flat work-list, and
    # coarsening it (Refinement 3b) amortizes the per-helper reconfig/init tax over
    # more tiles per call. D<=256 keep the fitted chunk; the D=128 perf shape (huge L1
    # headroom, 740 Q-blocks) coarsens to fill compute; D∈{512,1024} shrink to fit L1.
    sq_chunk_t, sk_chunk_t = _pick_chunks(
        sqt,
        skvt,
        dht,
        needs_mask_cb,
        in_df,
        out_df,
        interm_df,
        scaler_df,
        mask_df,
        bh=b * h_q,
        num_cores=num_cores,
    )
    q_num_chunks = sqt // sq_chunk_t
    k_num_chunks = skvt // sk_chunk_t

    # Subblock decomposition for the two matmuls (block held in DEST, num_k_blocks=1).
    qk_sb_h, qk_sb_w, qk_in0_sb, qk_in1_sb = _pick_subblock(sq_chunk_t, sk_chunk_t, dst_limit)
    pv_sb_h, pv_sb_w, pv_in0_sb, pv_in1_sb = _pick_subblock(sq_chunk_t, dht, dst_limit)

    total_q_blocks = b * h_q * q_num_chunks

    # ---- Refinement 3: K/V reuse-multicast gate (perf, scheme-change) ----------
    # K/V do not vary along S_q, so every core owning a Q-block of the same
    # (batch,head) re-reads the identical K/V from DRAM — the dominant bottleneck
    # on the flagged profile (~740 Q-blocks over 110 cores, ~2.4 MB K + 2.4 MB V
    # re-pulled per core). When the (batch,head) groups map exactly one-per-grid-row
    # (b·H_q == grid rows) and there is no mask (mask varies along S_q → not shared),
    # switch to a one-injector-per-row broadcast: col 0 of each row reads each
    # KV-block once and NoC-multicasts it across its row (ttnn.Mcast1D PerRow +
    # mcast_pipe). Every core in the row processes `rounds = ceil(q_num_chunks/GC)`
    # Q-blocks in perfect cb_k_in/cb_v_in lockstep (dummy slots re-run Q-chunk 0 —
    # a benign bit-identical redundant output), keeping the mcast landing address
    # identical across the row. All other cells keep the per-core DRAM path,
    # byte-identical to prior phases (USE_MCAST=0). No SUPPORTED change.
    #
    # PARKED (Refinement 3 outcome): the scheme is CORRECT but measured FLAT on the
    # flagged shape (11.05→10.97 ms; two ablations pin the shape as compute / per-core
    # dataflow-latency bound, NOT redundant-read bound — see changelog R3 and the
    # compute-side follow-up). Auto-firing it also exposed a rare intermittent
    # mcast-handshake hang that regressed the golden `test_op_loose` flagged case (the
    # completion-gate violation this debug pass fixes). Per "keep a correct lever at
    # its trivial byte-identical default as a live knob": the auto-gate is PARKED OFF
    # behind an explicit opt-in, so the whole supported rectangle (including the
    # flagged loose case) runs the proven, deterministic R2 per-core DRAM path
    # (byte-identical, zero hang, zero regression). The scheme stays fully intact and
    # is re-enabled — for any FUTURE genuinely read-bound shape — by exporting
    # TTNN_SDPA_KV_MCAST=1 (then the shape gate below applies as before).
    # Causal (needs_mask_cb) never takes the mcast path: the mask varies along S_q,
    # so K/V are not the only S_q-shared operand, and the mcast reader has no mask
    # branch. The gate already requires no mask; needs_mask_cb also excludes causal.
    mcast_opt_in = os.environ.get("TTNN_SDPA_KV_MCAST", "0") == "1"
    use_mcast = mcast_opt_in and (not needs_mask_cb) and (b * h_q == grid_rows) and (grid_cols > 1)

    semaphores = []
    mcast_ct = [0, 0, 0, 0, 0]  # McastArgs CT block (5 words); inert when !use_mcast

    if use_mcast:
        mcast_rounds = (q_num_chunks + grid_cols - 1) // grid_cols
        grid_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_cols - 1, grid_rows - 1))]
        )
        mc = ttnn.Mcast1D(
            device,
            grid_crs,
            ttnn.Mcast1DShape.PerRow,
            0,  # sender_index: col 0 injects each row
            ttnn.McastConfig(handshake=True, base_sem_id=0),
        )
        semaphores = mc.owned_semaphores()
        mcast_ct = list(mc.compile_time_args())
        all_cores = grid_crs

        # Per-core work: (core, q_start, q_count, row_y, col_x, rounds, is_sender, mcast_rt[4]).
        # q_start/q_count are unused on the mcast path (compute drives off `rounds`).
        assignment = []
        for y in range(grid_rows):
            for x in range(grid_cols):
                core = ttnn.CoreCoord(x, y)
                assignment.append(
                    (core, 0, 0, y, x, mcast_rounds, int(mc.is_sender(core)), list(mc.runtime_args(core)))
                )
    else:
        (
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            units_per_core_g1,
            units_per_core_g2,
        ) = ttnn.split_work_to_cores(grid, total_q_blocks)

        # Per-core flat work slice [q_start, q_start+q_count).
        assignment = []
        start = 0
        for group, per_core in ((core_group_1, units_per_core_g1), (core_group_2, units_per_core_g2)):
            if per_core == 0:
                continue
            for core in ttnn.corerange_to_cores(group, None, True):
                assignment.append((core, start, per_core, 0, 0, 0, 0, [0, 0, 0, 0]))
                start += per_core

    # ---- Circular buffers (built from the shared _cb_specs — single source) ----
    # Per-role dtype rationale (formats resolved above, before the block-factor pick):
    #   * input CBs (Q/K/V/mask) carry the user input dtype — the reader byte-
    #     copies DRAM tiles into them, so the CB tile size must match.
    #   * cb_out follows the input dtype (op contract; writer byte-copies out).
    #   * intermediates: fp32 ONLY for float32 INPUT. The online-softmax running
    #     (m,l,O) is parked in a CB and reloaded every KV-block; for a float32
    #     input the CB must carry fp32 or the park truncates the mantissa mid-
    #     reduce and erases the point of the fp32 datapath (coupled to the float32
    #     dtype, NOT to the bf16-input fp32_dest_acc_en=True config). bf16 / bf8b
    #     inputs keep bf16 intermediates: byte-identical to Phase-0. bf8b can't be
    #     an intermediate accumulator format; bf16 is the correct upcast target and
    #     already dominates the bf8b error budget.
    #   * scalers stay bf16 (reader packs them via prepare_reduce_scaler).
    def cb(index, num_pages, data_format):
        t = ttnn.tile_size(data_format)
        return ttnn.CBDescriptor(
            total_size=num_pages * t,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=t)],
        )

    cbs = [
        cb(index, num_pages, data_format)
        for index, num_pages, data_format in _cb_specs(
            sq_chunk_t, sk_chunk_t, dht, needs_mask_cb, in_df, out_df, interm_df, scaler_df, mask_df
        )
    ]

    # ---- Reader kernel ----
    reader_ct = [
        b,
        h_q,
        h_kv,
        sqt,
        skvt,
        dht,
        sq_chunk_t,
        sk_chunk_t,
        q_num_chunks,
        k_num_chunks,
        mask_regime,  # 0=none, 1=custom (DRAM read), 2=causal (on-device generate)
        mask_broadcast_head,
        1 if use_mcast else 0,
        grid_cols,
    ]
    reader_ct.extend(mcast_ct)  # McastArgs CT block -> reader CT [14..18]
    reader_ct.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    q_addr = query.buffer_address()
    k_addr = key.buffer_address()
    v_addr = value.buffer_address()
    m_addr = attn_mask.buffer_address() if has_mask else 0
    o_addr = output_tensor.buffer_address()
    for core, q_start, q_count, row_y, col_x, rnds, is_sender, mrt in assignment:
        # Reader RT: [q,k,v,m addrs | q_start,q_count | row_y,col_x,rounds,is_sender | mcast_rt(4)]
        reader_rt[core.x][core.y] = [
            q_addr,
            k_addr,
            v_addr,
            m_addr,
            q_start,
            q_count,
            row_y,
            col_x,
            rnds,
            is_sender,
        ] + mrt
        # Writer RT: [o_addr, q_start, q_count, row_y, col_x, rounds]
        writer_rt[core.x][core.y] = [o_addr, q_start, q_count, row_y, col_x, rnds]
        # Compute drives off block count: rounds (mcast) or the flat q_count slice.
        # q_start (RT[2]) lets causal recover each Q-block's global query-chunk index
        # (unused off the causal path; mcast never carries a mask so q_start=0 there).
        compute_rt[core.x][core.y] = [rnds if use_mcast else q_count, k_num_chunks, q_start]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Compute kernel ----
    # k_num_chunks is a RUNTIME arg (not CT): a constexpr loop bound would let
    # the compiler fully unroll the large kv_step body per KV-block, blowing the
    # kernel-config buffer for long sequences. Keeping it runtime keeps the loop
    # rolled (one kv_step instantiation per parity).
    compute_ct = [
        sq_chunk_t,
        sk_chunk_t,
        dht,
        mask_regime,  # 0=none, 1=custom, 2=causal (truncate KV loop + mask straddling blocks)
        _f32_bits(scale),
        qk_in0_sb,
        qk_in1_sb,
        qk_sb_h,
        qk_sb_w,
        pv_in0_sb,
        pv_in1_sb,
        pv_sb_h,
        pv_sb_w,
        int(os.environ.get("TTNN_SDPA_ABLATE", "0")),  # /perf-measure ablation gate (0=normal)
        # Refinement 3d — SFPU-floor lever (perf): route the compute config's
        # math_approx_mode into the SFPU exp datapath. The phase-4 exp over the whole
        # score block is the single dominant SFPU cost (ablation: 21%+ of the wall; fast
        # exp measured 1.44× — 10.25→7.12 ms — on the flagged shape). Fast exp trades a
        # little accuracy (flagged PCC 0.9997→0.9967), so it fires ONLY when the user
        # opts into approximate SFPU math via `math_approx_mode=True`. Default False →
        # exact exp → byte-identical to prior phases (zero regression on the exact path,
        # including the flagged perf test which requests math_approx_mode=False).
        1 if bool(getattr(compute_kernel_config, "math_approx_mode", False)) else 0,
        # Refinement 4 — causal: Q_NUM_CHUNKS lets compute recover each Q-block's
        # global query-chunk index (qc = (q_start + qb) % q_num_chunks) to truncate
        # the KV loop + gate the mask add identically to the reader.
        q_num_chunks,
    ]
    # Rebuild the compute config with the dtype-correct fidelity (never pass a
    # HiFi4 + fp32-DEST + bf16 combo through — issue #38306). fp32_dest_acc_en
    # and math_approx_mode are honored as-passed.
    resolved_compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=_resolve_math_fidelity(
            query.dtype, getattr(compute_kernel_config, "math_fidelity", ttnn.MathFidelity.HiFi2)
        ),
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=bool(getattr(compute_kernel_config, "math_approx_mode", False)),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=resolved_compute_config,
    )

    # ---- Writer kernel ----
    writer_ct = [b, h_q, sqt, dht, sq_chunk_t, q_num_chunks, 1 if use_mcast else 0, grid_cols]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=semaphores,  # Refinement 3: mcast data_ready + consumer_ready (empty otherwise)
        cbs=cbs,
    )

    ordered = [query, key, value] + ([attn_mask] if has_mask else []) + [output_tensor]
    return descriptor, ordered
