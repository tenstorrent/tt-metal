# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for Flash-Attention scaled_dot_product_attention.

Work distribution: one work item = one ``(batch b, query-head h_q, q-block qb)``
triple, producing one ``B_q x DHt`` output block by streaming all KV blocks with
an online softmax. Work items are split across the compute grid with
``split_work_to_cores``; each core decodes its flat work index range into
(b, h_q, qb) in the reader/writer kernels and runs the identical compute kernel.

Flash constraint (load-bearing): every score-bearing CB (cb_qk_scores, cb_p) is
sized to ONE ``B_q x B_kv`` block, never ``Sq_t x Sk_t``.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# Bound block heights so per-core L1 + score-CB footprint stays small. Both are
# always chosen as divisors of the respective tile-dim so tile-aligned inputs
# need no Q/K padding.
MAX_B_Q = 4
MAX_B_KV = 4

# L1-aware cap on the input-block footprint, in BYTES. cb_q_in / cb_k_in /
# cb_v_in are sized ~2 * B_{q,kv} * DHt tiles, so the per-core L1 footprint
# grows with B * DHt * (input tile bytes). For large head_dim (DHt = D/32) this
# overflows L1 (program.cpp CB allocation throw) at B_q = B_kv = 4. Cap
# B_{q,kv} so B_{q,kv} * DHt * tile_bytes stays under this budget; for small DHt
# (the common case, D<=128) this is inert and B stays at MAX_B_*; for D>=512 it
# pushes B down (to 1 at D=1024). A byte budget (vs a tile count) makes the cap
# dtype-aware: fp32 inputs (2x the bf16 page) get half the tile budget for the
# same L1 footprint. 32768 bytes == 16 bf16 tiles, reproducing the Phase-0 cap
# for bf16. This is block sizing, not a feature gate — the kernel is identical;
# only the per-core work granularity shrinks.
MAX_INPUT_BLOCK_BYTES = 32768


def _largest_divisor_leq(n: int, cap: int) -> int:
    for c in range(min(cap, n), 0, -1):
        if n % c == 0:
            return c
    return 1


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _f32_bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", x))[0]


# Default compute-config fields (used when the caller passes no
# compute_kernel_config). fp32_dest_acc_en is ON by default: the online-softmax
# recurrence and the QK / PV matmuls accumulate across the DHt K-loop and across
# every KV block, so fp32 dest accumulation is what lets the fp32 dtype deliver
# fp32-band precision (golden PCC>=0.999) and the bf16 Q1x1x128x1024
# explicit-scale cell clear its precision near-miss. HiFi4 maximizes the matmul
# pass count (full TF32 mantissa) and is what gives fp32's tight rel-RMS<=0.02
# tolerance real headroom (HiFi2 leaves fp32 marginal/failing on deep-D shapes).
#
# Verifier Phase-0 note flagged HiFi4 + fp32_dest_acc + bf16 matmul-path SUM
# reduce as the known-bad combo (issue #38306). Tested directly on this kernel
# (bf16 multi-KV, SUM reduce every block): PCC=1.00000, rel-RMS~0.0024 — NO
# corruption. The generic compute_kernel_lib::reduce threads MATH_FIDELITY as a
# template arg (it does NOT hardcode HiFi4), and the SUM-reduce intermediates
# here are fp32 (acc_dtype), so the bf16-SUM-reduce failure path of #38306 is
# not reached. The feared constraint does not bind for the helper-based reduce.
#
# Refinement-1 deliberately changes the Phase-0 default (HiFi2 / fp32_dest_acc
# OFF) — accuracy is this refinement's purpose; callers wanting throughput pass
# an explicit compute_kernel_config (e.g. HiFi2 / LoFi).
_DEFAULT_MATH_FIDELITY = ttnn.MathFidelity.HiFi4
_DEFAULT_FP32_DEST_ACC = True
_DEFAULT_MATH_APPROX = False
_DEFAULT_DST_FULL_SYNC = False


def _resolve_compute_config(compute_kernel_config):
    """Map an optional user ttnn.*ComputeKernelConfig onto the descriptor-level
    fields. None -> the Refinement-1 defaults above."""
    if compute_kernel_config is None:
        return (
            _DEFAULT_MATH_FIDELITY,
            _DEFAULT_FP32_DEST_ACC,
            _DEFAULT_MATH_APPROX,
            _DEFAULT_DST_FULL_SYNC,
        )
    return (
        getattr(compute_kernel_config, "math_fidelity", _DEFAULT_MATH_FIDELITY),
        bool(getattr(compute_kernel_config, "fp32_dest_acc_en", _DEFAULT_FP32_DEST_ACC)),
        bool(getattr(compute_kernel_config, "math_approx_mode", _DEFAULT_MATH_APPROX)),
        bool(getattr(compute_kernel_config, "dst_full_sync_en", _DEFAULT_DST_FULL_SYNC)),
    )


def create_program_descriptor(
    Q: ttnn.Tensor,
    K: ttnn.Tensor,
    V: ttnn.Tensor,
    attention_mask,
    output_tensor: ttnn.Tensor,
    *,
    scale: float,
    compute_kernel_config=None,
) -> ttnn.ProgramDescriptor:
    device = Q.device()

    math_fidelity, fp32_dest_acc_en, math_approx_mode, dst_full_sync_en = _resolve_compute_config(compute_kernel_config)

    # bfloat8_b requires fp32 dest accumulation: the block-float matmul datapath
    # produces uncorrelated output (PCC ~0.06, rel-RMS >5) with bf16 dest. There
    # is no valid bf16-dest mode for bf8b, so force fp32_dest_acc_en on
    # regardless of the caller's flag — preventing a silent-garbage footgun for a
    # user who passes a throughput-oriented config (fp32_dest_acc_en=False) with
    # bf8b inputs. fp32 and bf16 inputs honor the caller's flag.
    if Q.dtype == ttnn.bfloat8_b:
        fp32_dest_acc_en = True

    q_shape = list(Q.shape)
    k_shape = list(K.shape)
    B, H_q, S_q, D = q_shape
    H_kv, S_kv = k_shape[1], k_shape[2]

    # Tile counts use CEIL division: a non-tile-aligned dim (D, S_q, or S_kv not
    # a multiple of 32) still occupies a full last tile physically (TILE_LAYOUT
    # pads up to the tile boundary). Floor division would silently drop the
    # partial last tile and only process the aligned prefix. For aligned dims
    # ceil == floor, so this is a no-op for tile_aligned inputs.
    DHt = _ceil_div(D, TILE_DIM)
    vDHt = DHt  # V head dim == D
    Sq_t = _ceil_div(S_q, TILE_DIM)
    Sk_t = _ceil_div(S_kv, TILE_DIM)

    # Number of valid (logical) columns in the partial LAST tile of each dim.
    # 0 means the dim is tile-aligned (no partial tile). kv_partial is the one
    # that the softmax-denominator masking cares about: the padded KV columns of
    # the last score tile-column must be set to -inf before the row-max / row-sum
    # (otherwise their score-0 contributes exp(0 - m) to the denominator).
    kv_partial = S_kv % TILE_DIM

    # Intermediate / scratch CB working format. With fp32 dest accumulation the
    # online-softmax running stats (m, l, O) and the score CBs must be parked in
    # Float32 between phases, otherwise pack_tile() truncates the fp32 dest back
    # to bf16 at every phase boundary and the fp32-acc gain is erased (skill §4).
    #
    # The intermediate format is ALSO forced to Float32 whenever the input dtype
    # is not bf16 (fp32 or bf8b), independent of fp32_dest_acc_en. Empirically,
    # bf16 intermediates fed from a non-bf16 input CB through the matmul→reduce→
    # eltwise helper chain produce uncorrelated output (PCC ~0.07-0.37, rel-RMS
    # >1) when fp32_dest_acc is off — the bf16 intermediate cannot carry the
    # non-bf16 input's matmul result correctly. Float32 intermediates fix it (and
    # cost nothing for fp32 input, which is already fp32-wide). So bf16
    # intermediates are used ONLY for the bf16-input + fp32_dest_acc-off case,
    # the one combination where input and intermediate formats match. bf8b is
    # never used for intermediates (block-float running stats would be lossy).
    input_is_bf16 = Q.dtype == ttnn.bfloat16
    use_bf16_intermediate = (not fp32_dest_acc_en) and input_is_bf16
    acc_dtype = ttnn.bfloat16 if use_bf16_intermediate else ttnn.float32
    acc_tile = ttnn.tile_size(acc_dtype)

    # KV-column padding mask: needed only when S_kv is not tile-aligned. The
    # padded KV columns of the last score tile-column score 0 (K's padded rows
    # are zero), and exp(0 - rowmax) would pollute the softmax denominator. The
    # reader generates a persistent additive -inf mask block (in acc_dtype, to
    # match the score CB so the binary add is same-format) and the compute adds
    # it on the LAST kv block before the row-max. kv_partial==0 (tile-aligned
    # S_kv) compiles the whole path out — zero impact on aligned inputs.
    need_pad_mask = kv_partial != 0
    pad_mask_is_fp32 = 1 if acc_dtype == ttnn.float32 else 0

    # L1-aware block cap: keep the per-core input-block footprint bounded in
    # BYTES (MAX_INPUT_BLOCK_BYTES). The input-block CBs (cb_q/k/v_in) carry the
    # input dtype, so an fp32 input (2x the bf16 page) automatically halves the
    # tile budget and pushes B down relative to bf16 — keeping the dominant CB
    # footprint constant across dtypes. MAX_INPUT_BLOCK_BYTES is set so bf16
    # reproduces the Phase-0 16-tile cap exactly (no bf16 behavior change). The
    # fp32 scratch (cb_o_*, cb_qk, ...) is small next to the 3x input CBs and
    # fits L1 comfortably at this budget.
    input_tile = int(Q.buffer_page_size())
    max_block_tiles = max(1, MAX_INPUT_BLOCK_BYTES // input_tile)
    dht_cap = max(1, max_block_tiles // DHt)
    B_q = _largest_divisor_leq(Sq_t, min(MAX_B_Q, dht_cap))
    B_kv = _largest_divisor_leq(Sk_t, min(MAX_B_KV, dht_cap))
    n_q = Sq_t // B_q
    n_kv = Sk_t // B_kv

    has_mask = attention_mask is not None
    mask_H = int(attention_mask.shape[1]) if has_mask else 1

    scale_bits = _f32_bits(scale)

    # --- page sizes (all bf16 tiles) ---
    q_page = Q.buffer_page_size()
    k_page = K.buffer_page_size()
    v_page = V.buffer_page_size()
    out_page = output_tensor.buffer_page_size()
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    mask_page = attention_mask.buffer_page_size() if has_mask else bf16_tile

    # --- work distribution ---
    total_work = B * H_q * n_q
    grid_size = device.compute_with_storage_grid_size()
    (
        num_cores,
        core_grid,
        core_group_1,
        core_group_2,
        items_g1,
        items_g2,
    ) = ttnn.split_work_to_cores(grid_size, total_work)

    # --- CB indices (semantic names) ---
    CB_Q = 0
    CB_K = 1
    CB_V = 2
    CB_MASK = 3
    CB_PAD_MASK = 4
    CB_MAX_SCALER = 8
    CB_SUM_SCALER = 9
    CB_ALPHA = 10
    CB_L_RECIP = 11
    CB_M_BLK = 12
    CB_OUT = 16
    CB_QK = 24
    CB_P = 25
    CB_O_BLK = 26
    CB_O_RUN = 27
    CB_M_PREV = 28
    CB_M_RUN = 29
    CB_L_RUN = 30
    CB_L_BLK = 31

    qk_tiles = B_q * B_kv
    o_tiles = B_q * vDHt

    def cb(index, page_size, num_pages, dtype=ttnn.bfloat16):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        # reader-facing inputs: double-buffered for reader/compute pipelining.
        # Format follows the input tensor dtype (bf16 / fp32 / bf8b).
        cb(CB_Q, q_page, 2 * B_q * DHt, Q.dtype),
        cb(CB_K, k_page, 2 * B_kv * DHt, K.dtype),
        cb(CB_V, v_page, 2 * B_kv * vDHt, V.dtype),
        # persistent reduce scalers: bf16-packed by prepare_reduce_scaler in the
        # reader regardless of input dtype — keep bf16.
        cb(CB_MAX_SCALER, bf16_tile, 1),
        cb(CB_SUM_SCALER, bf16_tile, 1),
        # per-iteration / persistent compute scratch (single-buffered: sequential).
        # acc_dtype = Float32 when fp32_dest_acc_en, else bf16 — these park the
        # online-softmax running stats / score block between phases.
        cb(CB_ALPHA, acc_tile, B_q, acc_dtype),
        cb(CB_L_RECIP, acc_tile, B_q, acc_dtype),
        cb(CB_M_BLK, acc_tile, B_q, acc_dtype),
        cb(CB_QK, acc_tile, qk_tiles, acc_dtype),
        cb(CB_P, acc_tile, qk_tiles, acc_dtype),
        cb(CB_O_BLK, acc_tile, o_tiles, acc_dtype),
        cb(CB_O_RUN, acc_tile, o_tiles, acc_dtype),
        cb(CB_M_PREV, acc_tile, B_q, acc_dtype),
        cb(CB_M_RUN, acc_tile, B_q, acc_dtype),
        cb(CB_L_RUN, acc_tile, B_q, acc_dtype),
        cb(CB_L_BLK, acc_tile, B_q, acc_dtype),
        # output: double-buffered for compute/writer pipelining. Format follows
        # the output tensor dtype.
        cb(CB_OUT, out_page, 2 * o_tiles, output_tensor.dtype),
    ]
    if has_mask:
        cbs.append(cb(CB_MASK, mask_page, 2 * qk_tiles, attention_mask.dtype))
    if need_pad_mask:
        # Persistent additive KV-column pad mask, one B_q x B_kv block in
        # acc_dtype. Generated once by the reader at startup, held (never popped)
        # by the compute. Single-buffered — it is constant across all work items.
        cbs.append(cb(CB_PAD_MASK, acc_tile, qk_tiles, acc_dtype))

    # ----------------------------------------------------------------------
    # Reader kernel
    # ----------------------------------------------------------------------
    reader_ct_args = [
        B,  # 0
        H_q,  # 1
        H_kv,  # 2
        Sq_t,  # 3
        Sk_t,  # 4
        DHt,  # 5
        vDHt,  # 6
        B_q,  # 7
        B_kv,  # 8
        n_q,  # 9
        n_kv,  # 10
        1 if has_mask else 0,  # 11
        mask_H,  # 12
        kv_partial,  # 13 (S_kv % 32; 0 => tile-aligned, no pad mask)
        pad_mask_is_fp32,  # 14 (acc_dtype fp32 => fp32 -inf fill, else bf16)
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(Q).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(K).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(V).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(attention_mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # ----------------------------------------------------------------------
    # Compute kernel
    # ----------------------------------------------------------------------
    compute_ct_args = [
        B_q,  # 0
        B_kv,  # 1
        DHt,  # 2
        vDHt,  # 3
        n_kv,  # 4
        1 if has_mask else 0,  # 5
        scale_bits,  # 6
        kv_partial,  # 7 (S_kv % 32; 0 => no pad mask add)
    ]

    # ----------------------------------------------------------------------
    # Writer kernel
    # ----------------------------------------------------------------------
    writer_ct_args = [
        H_q,  # 0
        Sq_t,  # 1
        DHt,  # 2 (== vDHt for output)
        B_q,  # 3
        n_q,  # 4
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ----------------------------------------------------------------------
    # Per-core runtime args
    # ----------------------------------------------------------------------
    q_addr = Q.buffer_address()
    k_addr = K.buffer_address()
    v_addr = V.buffer_address()
    mask_addr = attention_mask.buffer_address() if has_mask else 0
    out_addr = output_tensor.buffer_address()

    reader_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()

    start = 0
    for group, items in ((core_group_1, items_g1), (core_group_2, items_g2)):
        if items == 0:
            continue
        for cr in group.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    reader_rt[x][y] = [start, items, q_addr, k_addr, v_addr, mask_addr]
                    compute_rt[x][y] = [items]
                    writer_rt[x][y] = [start, items, out_addr]
                    start += items

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    # Compute config: caller-driven via compute_kernel_config (resolved above).
    # No UnpackToDestFp32 tags — every intermediate CB feeds an FPU op (matmul /
    # reduce / FPU binary), which is incompatible with UnpackToDestFp32 (skill
    # §1.5); the precision lever here is fp32 dest accumulation + fp32 scratch
    # CBs, not lossless reload.
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_config,
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
