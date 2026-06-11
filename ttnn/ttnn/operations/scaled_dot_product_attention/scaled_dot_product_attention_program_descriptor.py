# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for scaled_dot_product_attention (Flash Attention).

Work unit = one (b, h, q_chunk): c_q tile-rows of Q against all of K/V.
Each core independently owns a contiguous range of flattened
(b*H + h)*Nq + q_chunk indices. Per-block CBs are sized c_q x c_kv tiles,
independent of S_kv (the load-bearing Flash Attention constraint).
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32

# --- CB indices (semantic names; numeric slot ranges: 0-7 in, 8-15 special,
# 16-23 out, 24-31 intermediate) ---
CB_Q_TILES = 0
CB_KT_TILES = 1
CB_V_TILES = 2
CB_MASK_TILES = 3
CB_SCALER_MAX = 8
CB_SCALER_SUM = 9
CB_CUR_SUM = 10
CB_PREV_MAX = 11
CB_RUNNING_MAX = 12
CB_ALPHA = 13
CB_RUNNING_SUM = 14
CB_INV_SUM = 15
CB_OUT_TILES = 16
CB_SCORES = 24
CB_SCORES_SCALED = 25
CB_PROBS = 26
CB_PV = 27
CB_O_ACC = 28
CB_CUR_MAX_FULL = 29  # block rowmax bcast to full tile (fp32, Refinement 5)
CB_M_FULL = 30  # running max as full tile (fp32, Refinement 5)
CB_PAD_MASK = 4  # S_kv pad mask row (bf16, Refinement 4): 0 valid / -1e9 pad cols


def _float_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]


def _flatten_cores(grid_size):
    return [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]


def create_program_descriptor(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    mask: ttnn.Tensor,
    output: ttnn.Tensor,
    *,
    scale: float,
    compute_kernel_config,
) -> ttnn.ProgramDescriptor:
    # Tile counts come from the PADDED shape (TILE layout pads the last two dims
    # to 32 with zeros). Logical shapes drive only validate()/scale, both handled
    # by the entry point. For non-tile-aligned S_kv (Refinement 4) the zero-padded
    # K rows produce score columns of 0 — they would corrupt rowmax (MAX sees 0)
    # and rowsum (exp(0 - m) != 0), so the reader prepares a pad-mask row (cb 4):
    # 0 in valid columns, -1e9 in the last tile's pad columns, added to the scores
    # before the running-max update. Padded D / S_q need no fix: zero D columns
    # add 0 to QK^T and 0 to P@V; pad Q rows produce garbage rows sliced off host-side.
    B, H = list(q.shape)[:2]
    _, S_q, D = list(q.padded_shape)[1:]
    _, H_kv, S_kv, _ = list(k.padded_shape)

    Dt = D // TILE
    Sq_t = S_q // TILE
    Skv_t = S_kv // TILE
    kv_rem = list(k.shape)[-2] % TILE  # valid cols in the last KV tile (0 = aligned)

    # Chunk size: c = clamp(16 / Dt, 1, 4)
    c = max(1, min(4, 16 // Dt))
    # KV is re-read per Q chunk, so DRAM traffic ~ 1/c_q and the profile shows
    # all favorable cases are DRAM-read-bound at small c_q. Use the biggest Q
    # chunk L1 allows: 8 tile-rows at Dt<=4, else 4.
    c_q = min(8 if Dt <= 4 else 4, Sq_t)
    # c_kv decoupled from the 16/Dt clamp (Dt=16 otherwise streams KV one
    # tile-row at a time, and halving it doubles online-softmax steps, which
    # compounds bf16 accumulator error below the 0.99 PCC bar at S=16k).
    # At d=64 the kernel is compute-leaning: wider KV blocks amortize the
    # per-block softmax/O-acc passes (L1 fits at Dt<=2 with c_q=8).
    c_kv = min(16 if Dt <= 2 else 4, Skv_t)
    Nq = -(-Sq_t // c_q)
    Nkv = -(-Skv_t // c_kv)
    c_q_last = Sq_t - (Nq - 1) * c_q
    c_kv_last = Skv_t - (Nkv - 1) * c_kv

    # P@V subblock width: largest divisor of Dt that is <= 4 (fp32 DEST limit).
    sw = 1
    for cand in (4, 3, 2, 1):
        if Dt % cand == 0:
            sw = cand
            break
    n_sub_w = Dt // sw

    has_mask = mask is not None
    mask_is_per_head = has_mask and list(mask.shape)[1] == H

    # Input/output CB formats follow the input dtype (validate() enforces
    # Q/K/V/mask dtype equality, and output dtype == query dtype). Stat /
    # accumulator intermediates AND cb_probs follow fp32_dest_acc_en: Float32
    # when the fp32 DEST accumulation crosses those CBs (default), Float16_b
    # otherwise — packing a 16-bit DEST into Float32 CBs corrupts values
    # (probe_007: pcc ~0.41 at every fidelity). cb_probs at input dtype was the
    # Refinement 3 long-context precision miss: quantized probs made rowsum l
    # and P@V inconsistent (probe_012). Reduce-scaler tiles stay bfloat16.
    in_fmt = q.dtype
    t_in = ttnn.tile_size(in_fmt)
    t_bf = ttnn.tile_size(ttnn.bfloat16)
    acc_fmt = ttnn.float32 if compute_kernel_config.fp32_dest_acc_en else ttnn.bfloat16
    t_acc = ttnn.tile_size(acc_fmt)

    # --- Work distribution over flattened (b, h, q_chunk) units ---
    total_units = B * H * Nq
    grid_size = q.device().compute_with_storage_grid_size()
    cores = _flatten_cores(grid_size)
    num_cores = min(len(cores), total_units)
    base = total_units // num_cores
    rem = total_units % num_cores

    full_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    # --- Circular buffers ---
    def cb(index, pages, page_size, fmt):
        return ttnn.CBDescriptor(
            total_size=pages * page_size,
            core_ranges=full_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)],
        )

    # Fast path (mask-free, aligned, uniform chunks) never touches the
    # online-softmax stat CBs — shrink them to 1 page to fund bigger chunks.
    use_fast = not has_mask and kv_rem == 0 and c_q_last == c_q and c_kv_last == c_kv
    stat = (lambda pages: 1) if use_fast else (lambda pages: pages)

    # --- KV multicast (perf-juice) ---
    # The deal-out is contiguous, so the g = Nq/count cores that share a head
    # consume the *same* KV stream. When g in {2,4,8} the group lies in one
    # grid row -> leader reads each KV block from DRAM once and row-mcasts it
    # to the followers, cutting KV DRAM traffic by g.
    count = total_units // num_cores
    g = (Nq // count) if (count and rem == 0 and Nq % count == 0) else 0
    use_mcast = not has_mask and kv_rem == 0 and H_kv == H and g in (2, 4, 8) and num_cores == len(cores)

    # Ones-column rowsum (fast path, small Dt): V rows carry a leading
    # all-ones tile, so the PV matmul's L1-acc also accumulates l in output
    # column 0 — the rowsum reduce disappears. V width becomes Dt+1.
    # Requires the writer-produced V (mcast mode).
    # Worth it where rowsum cost exceeds the extra PV column (Dt in 3..4 with
    # mcast; at Dt=2 the +50% PV width and writer-side V read lose).
    ones_col = use_fast and use_mcast and Dt in (3, 4)
    v_w = Dt + 1 if ones_col else Dt

    cbs = [
        cb(CB_Q_TILES, c_q * Dt, t_in, in_fmt),
        cb(CB_KT_TILES, 2 * c_kv * Dt, t_in, in_fmt),
        cb(CB_V_TILES, 2 * c_kv * v_w, t_in, in_fmt),
        cb(CB_SCALER_MAX, 1, t_bf, ttnn.bfloat16),
        cb(CB_SCALER_SUM, 1, t_bf, ttnn.bfloat16),
        cb(CB_CUR_SUM, c_q, t_acc, acc_fmt),
        # 2x: Phase 5b pushes the new m_prev before popping the old block.
        cb(CB_PREV_MAX, stat(2 * c_q), t_acc, acc_fmt),
        cb(CB_RUNNING_MAX, stat(2 * c_q), t_acc, acc_fmt),
        cb(CB_ALPHA, stat(c_q), t_acc, acc_fmt),
        cb(CB_RUNNING_SUM, stat(2 * c_q), t_acc, acc_fmt),
        cb(CB_INV_SUM, c_q, t_acc, acc_fmt),
        # Single-buffered at large Dt to fit c_q=4 + c_kv=4 in L1 (writer
        # overlap is negligible there — the case is DRAM-read-bound).
        cb(CB_OUT_TILES, (1 if Dt > 8 else 2) * c_q * Dt, t_in, in_fmt),
        # Fast path packs exp(QK^T) straight to cb_probs; cb_scores unused.
        cb(CB_SCORES, 1 if use_fast else c_q * c_kv, t_acc, acc_fmt),
        # Fast path repurposes cb_scores_scaled to hold the pre-scaled Q chunk
        # (c_q * Dt tiles, retained across the unit's KV loop).
        cb(CB_SCORES_SCALED, max(c_q * c_kv, c_q * Dt), t_acc, acc_fmt),
        cb(CB_PROBS, c_q * c_kv, t_acc, acc_fmt),
        # Fast path: PV matmul targets Interm (cb_o_acc); cb_pv is an unread
        # out placeholder.
        cb(CB_PV, 1 if use_fast else c_q * v_w, t_acc, acc_fmt),
        # Fast path: single block — the PV L1-acc spill must land at the SAME
        # L1 offsets every K-block; a 2-block FIFO alternates regions and
        # splits the accumulation. Slow path: 2 blocks (same-CB read/write).
        cb(CB_O_ACC, (1 if use_fast else 2) * c_q * v_w, t_acc, acc_fmt),
        cb(CB_CUR_MAX_FULL, stat(c_q), t_acc, acc_fmt),
        cb(CB_M_FULL, stat(c_q), t_acc, acc_fmt),
    ]
    if has_mask:
        cbs.append(cb(CB_MASK_TILES, 2 * c_q * c_kv, t_in, in_fmt))
    if kv_rem != 0:
        # Pad-mask row: c_kv_last bf16 tiles (zeros except the last tile's pad
        # cols at -1e9), prepared once by the reader, never popped (HeldBulk).
        cbs.append(cb(CB_PAD_MASK, c_kv_last, t_bf, ttnn.bfloat16))

    # --- Reader ---
    reader_ct_args = [
        H,
        Sq_t,
        Skv_t,
        Dt,
        c_q,
        c_kv,
        Nq,
        Nkv,
        c_q_last,
        c_kv_last,
        1 if has_mask else 0,
        1 if mask_is_per_head else 0,
        H_kv,  # GQA/MQA: Q head h -> KV head h / (H // H_kv); == H for MHA
        kv_rem,  # valid cols in the last KV tile (0 = S_kv tile-aligned)
        g if use_mcast else 0,  # KV mcast group size (0 = off)
        1 if (use_mcast or ones_col) else 0,  # writer owns V
        0,  # reader-V mcast disabled (NoC0 saturates; V rides NoC1 in writer)
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(q).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(k).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(v).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # --- Writer (also produces cb_v on NoC1 in mcast mode) ---
    writer_ct_args = [
        H,
        Sq_t,
        Dt,
        c_q,
        Nq,
        c_q_last,
        Skv_t,
        c_kv,
        Nkv,
        g if use_mcast else 0,
        1 if ones_col else 0,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(v).get_compile_time_args())

    # --- Compute ---
    compute_ct_args = [
        Dt,
        c_q,
        c_kv,
        Nq,
        Nkv,
        c_q_last,
        c_kv_last,
        sw,
        n_sub_w,
        1 if has_mask else 0,
        # CT arg 10 reserved (out-is-bf16): kernel no longer branches on it —
        # the packer rounds half-up, no DEST rounding pass needed (probe_017/019).
        1 if in_fmt == ttnn.bfloat16 else 0,
        1 if kv_rem != 0 else 0,  # HAS_PAD: add pad-mask row on the last KV block
        1 if ones_col else 0,  # ONES_COL: V carries a leading ones tile per row
    ]

    scale_bits = _float_bits(scale)

    device = q.device()
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    start = 0
    for i, core in enumerate(cores):
        cnt = (base + (1 if i < rem else 0)) if i < num_cores else 0
        reader_args = [
            q.buffer_address(),
            k.buffer_address(),
            v.buffer_address(),
            mask.buffer_address() if has_mask else 0,
            start,
            cnt,
        ]
        writer_args = [output.buffer_address(), start, cnt]
        if use_mcast:
            # Group = g consecutive cores in this row. Leader on the diagonal
            # (x = row mod g) so 8 leaders don't pile DRAM traffic on one column.
            gi = (i // g) * g
            leader = cores[gi + (cores[gi].y % g)]
            p_first = device.worker_core_from_logical_core(cores[gi])
            p_last = device.worker_core_from_logical_core(cores[gi + g - 1])
            p_leader = device.worker_core_from_logical_core(leader)
            mcast_args = [
                1 if core == leader else 0,
                p_first.x,
                p_first.y,
                p_last.x,
                p_last.y,
                p_leader.x,
                p_leader.y,
            ]
            reader_args += mcast_args
            writer_args += [v.buffer_address()] + mcast_args
        elif ones_col:
            writer_args += [v.buffer_address()]
        reader_rt[core.x][core.y] = reader_args
        writer_rt[core.x][core.y] = writer_args
        compute_rt[core.x][core.y] = [start, cnt, scale_bits]
        start += cnt

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=full_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=full_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
        math_approx_mode=compute_kernel_config.math_approx_mode,
        dst_full_sync_en=compute_kernel_config.dst_full_sync_en,
    )

    # Per-CB unpack-to-DEST mode. With the framework default (UnpackToDestMode.Default)
    # a Float32 CB is unpacked through the 16-bit SrcA/SrcB datapath, so copy_tile of an
    # fp32 CB into DEST silently truncates fp32 -> fp16 (base_types.hpp: "Default mode
    # enables all dataformats EXCEPT Float32 to be unpacked into Dest"). That truncated
    # exactly the P@V accumulator O (cb_pv fp32-exact, but cb_o_acc held O at fp16
    # precision), the dominant single-bf16-ulp flip source on near-uniform attention
    # (Refinement 5). UnpackToDestFp32 on every Float32 CB makes the whole stat/accumulator
    # copy_tile path fp32-exact, as the kernel's design comment intends. The bf16 input
    # CBs (Q/K/V/mask, scaler tiles) stay Default — only acc_fmt==Float32 CBs flip.
    if acc_fmt == ttnn.float32:
        NUM_CBS = 32
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        fp32_cbs = [
            CB_PV,
            CB_O_ACC,
            CB_CUR_SUM,
            CB_PREV_MAX,
            CB_RUNNING_MAX,
            CB_ALPHA,
            CB_RUNNING_SUM,
            CB_INV_SUM,
            CB_SCORES,
            CB_CUR_MAX_FULL,
            CB_M_FULL,
        ]
        for idx in fp32_cbs:
            modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = modes

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=full_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_config,
    )

    # Sems 0/1: K^T handshake on the reader (NoC0); sems 2/3: V handshake on
    # the writer (NoC1). 0/2 = followers ready (count to g-1), 1/3 = landed.
    semaphores = (
        [ttnn.SemaphoreDescriptor(id=i, core_ranges=full_grid, initial_value=0) for i in range(4)] if use_mcast else []
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
