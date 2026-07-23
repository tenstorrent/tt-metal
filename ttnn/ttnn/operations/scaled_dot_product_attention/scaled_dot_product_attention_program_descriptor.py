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
Q_CHUNK_TILES = 4  # Sq_chunk_t upper bound (tiles of 32 query rows)
K_CHUNK_TILES = 4  # Sk_chunk_t upper bound (tiles of 32 key/value rows)
KV_BUFFER_FACTOR = 2  # double-buffer K/V/mask to overlap DRAM read with compute
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


def _cb_specs(sq, sk, dht, has_mask, in_df, out_df, interm_df, scaler_df):
    """(index, num_pages, data_format) for every CB — the SINGLE source of truth
    shared by the L1-footprint budget calc and the actual CB build below. Page
    counts are functions of the block factors sq/sk and DHt (never the full S)."""
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
    if has_mask:
        specs.append((CB_MASK_IN, qk_tiles * KV_BUFFER_FACTOR, in_df))
    return specs


def _cb_footprint_bytes(specs):
    return sum(num_pages * ttnn.tile_size(df) for _, num_pages, df in specs)


def _pick_chunks(sqt, skvt, dht, has_mask, in_df, out_df, interm_df, scaler_df):
    """Coarsest (largest-footprint) block-factor pair (sq_chunk_t, sk_chunk_t)
    whose per-core CB footprint fits the L1 budget. When the design-default pair
    (Q_CHUNK_TILES, K_CHUNK_TILES) already fits (all D<=256 and the D=128 perf
    shape), it is the max-footprint candidate and is chosen unchanged — so no
    currently-passing cell regresses. Falls back to (1,1) if even that OOMs
    (D-blocking is the next refinement's scope, not reached for D<=1024)."""
    best = None  # (footprint, sq, sk)
    for sq in _divisors_leq(sqt, Q_CHUNK_TILES):
        for sk in _divisors_leq(skvt, K_CHUNK_TILES):
            fp = _cb_footprint_bytes(_cb_specs(sq, sk, dht, has_mask, in_df, out_df, interm_df, scaler_df))
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


def create_program_descriptor(query, key, value, attn_mask, output_tensor, *, scale, compute_kernel_config):
    b, h_q, s_q, d = tuple(query.shape)
    _, h_kv, s_kv, _ = tuple(key.shape)

    sqt = s_q // 32
    skvt = s_kv // 32
    dht = d // 32

    has_mask = attn_mask is not None
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

    # ---- L1-budget cap on the block factors (Refinement 2) ----
    # sq_chunk_t/sk_chunk_t are the design's block-factor knobs; pick the coarsest
    # divisor pair whose CB footprint fits L1. Knob-turn only — the kernel is
    # already parameterized on (sq, sk, dht); shrinking the chunk just adds more
    # (fully-full) Q/KV chunks to the flat work-list. D<=256 / the D=128 perf shape
    # keep the (4,4) default (byte-identical to Phase-0); D∈{512,1024} shrink to fit.
    sq_chunk_t, sk_chunk_t = _pick_chunks(sqt, skvt, dht, has_mask, in_df, out_df, interm_df, scaler_df)
    q_num_chunks = sqt // sq_chunk_t
    k_num_chunks = skvt // sk_chunk_t

    # Subblock decomposition for the two matmuls (block held in DEST, num_k_blocks=1).
    qk_sb_h, qk_sb_w, qk_in0_sb, qk_in1_sb = _pick_subblock(sq_chunk_t, sk_chunk_t, dst_limit)
    pv_sb_h, pv_sb_w, pv_in0_sb, pv_in1_sb = _pick_subblock(sq_chunk_t, dht, dst_limit)

    total_q_blocks = b * h_q * q_num_chunks

    device = query.device()
    grid = device.compute_with_storage_grid_size()
    grid_cols, grid_rows = grid.x, grid.y

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
    mcast_opt_in = os.environ.get("TTNN_SDPA_KV_MCAST", "0") == "1"
    use_mcast = mcast_opt_in and (not has_mask) and (b * h_q == grid_rows) and (grid_cols > 1)

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
            sq_chunk_t, sk_chunk_t, dht, has_mask, in_df, out_df, interm_df, scaler_df
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
        1 if has_mask else 0,
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
        compute_rt[core.x][core.y] = [rnds if use_mcast else q_count, k_num_chunks]

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
        1 if has_mask else 0,
        _f32_bits(scale),
        qk_in0_sb,
        qk_in1_sb,
        qk_sb_h,
        qk_sb_w,
        pv_in0_sb,
        pv_in1_sb,
        pv_sb_h,
        pv_sb_w,
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
