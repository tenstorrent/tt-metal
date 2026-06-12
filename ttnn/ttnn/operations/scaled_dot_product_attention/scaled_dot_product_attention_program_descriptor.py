# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for scaled_dot_product_attention (Flash Attention).

Work unit = one (batch b, head h, Q-block i) triple producing one
q_chunk_t x D_t output block. q_chunk_t == k_chunk_t == 1 tile: each work
unit reads one tile-row of Q (D_t tiles), iterates over S_kv_t KV blocks
(one tile-row of K/V each), and runs the online-softmax recurrence in the
compute kernel. The full S_q x S_kv score matrix is never materialized —
the score / prob CBs are a single tile.

Distribution: total_units = B * H * S_q_t spread contiguously across the
compute grid; each core gets [start_unit, num_units).
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# CB indices (must match the kernels).
CB_Q_IN = 0
CB_K_IN = 1
CB_V_IN = 2
CB_MASK_IN = 3
CB_SCALE = 8
CB_SCALER_MAX = 9
CB_SCALER_SUM = 15
CB_MAX = 10
CB_MAX_PREV = 11
CB_CORR = 12
CB_L = 13
CB_L_BLOCK = 14
CB_M_BLK = 23
CB_QK = 24
CB_P = 25
CB_O_ACC = 26
CB_PV = 27
CB_O_TMP = 28
CB_OUT = 16


def _f32_bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", float(x)))[0]


def create_program_descriptor(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attn_mask,
    output_tensor: ttnn.Tensor,
    *,
    scale: float,
    compute_kernel_config=None,
) -> ttnn.ProgramDescriptor:
    device = query.device()

    # --- Resolve compute config ------------------------------------------------
    # Defaults reproduce the Phase-0 hard-coded ComputeConfigDescriptor exactly:
    # HiFi2 (NOT HiFi4 — known-bad with bf16 + fp32_dest_acc on Wormhole B0) and
    # fp32 DEST accumulation for the online-softmax recurrence. A caller passing
    # nothing sees byte-identical behavior to Phase 0.
    if compute_kernel_config is not None:
        math_fidelity = compute_kernel_config.math_fidelity
        fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en
        math_approx_mode = compute_kernel_config.math_approx_mode
        dst_full_sync_en = compute_kernel_config.dst_full_sync_en
    else:
        math_fidelity = ttnn.MathFidelity.HiFi2
        fp32_dest_acc_en = True
        math_approx_mode = False
        dst_full_sync_en = False

    b, h, s_q, d = (int(x) for x in query.shape)
    s_kv = int(key.shape[-2])
    mask_h = int(attn_mask.shape[1]) if attn_mask is not None else 1
    has_mask = attn_mask is not None

    D_t = d // TILE_DIM
    S_q_t = s_q // TILE_DIM
    S_kv_t = s_kv // TILE_DIM

    total_units = b * h * S_q_t

    # --- Work distribution over the compute grid ---
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = min(total_units, max_cores)
    cores = ttnn.grid_to_cores(num_cores, grid.x, grid.y, True)

    base = total_units // num_cores
    rem = total_units % num_cores

    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    # --- Circular-buffer data formats (derived from dtype + compute config) ---
    # ttnn dtypes double as CB data formats here. Three roles:
    #   * input-side  (Q/K/V/mask) → the input tensor dtype (bf16/fp32/bf8b).
    #   * output-side (cb_out)     → the output tensor dtype.
    #   * intermediate / accumulator (running m_i, l_i, O_i, their scratch, AND
    #     the score/prob blocks) → fp32 when fp32_dest_acc_en, else input dtype.
    # The online-softmax recurrence keeps its running accumulators in fp32:
    # packing them back to a lower-precision format between KV blocks compounds
    # rounding across the KV loop (error grows with S / num_kv_blocks) — see
    # op_design.md Key Risks ("Numerical exactness requires fp32 DEST
    # accumulation"). Keeping the score/prob blocks (cb_qk/cb_p) at fp32 too is
    # the score-path precision lever folded in from the verifier's deferred
    # observation #2 (lifts the sign-biased / low-variance canaries). This
    # replaces the Phase-0 hard-coded f32 accumulator formats: when a caller
    # turns fp32_dest_acc_en off, the intermediates follow the input dtype.
    #
    # No CB is tagged UnpackToDestFp32: every intermediate/accumulator CB feeds
    # at least one FPU op (matmul / reduce / FPU binary), and an
    # UnpackToDestFp32-tagged CB cannot participate in any FPU op (it bypasses
    # srcA/srcB). The fp32 storage already gives the precision win; the FPU
    # inputs land in TF32 regardless, which is the unavoidable srcA/srcB drop.
    input_fmt = query.dtype
    out_fmt = output_tensor.dtype
    if fp32_dest_acc_en:
        accum_fmt = ttnn.float32
    elif input_fmt == ttnn.bfloat8_b:
        # Block-float (16 values share an exponent) is unusable for the online-
        # softmax running stats: cb_max / cb_l hold one valid value per row and
        # cb_qk / cb_p hold raw scores / probabilities — bf8b storage collapses
        # them (PCC -> 0). Floor bf8b intermediates to bf16 when fp32 DEST acc is
        # off. (With the default fp32_dest_acc_en=True this branch is dead.)
        accum_fmt = ttnn.bfloat16
    else:
        accum_fmt = input_fmt

    def cb(index, num_pages, fmt=input_fmt):
        page = ttnn.tile_size(fmt)
        return ttnn.CBDescriptor(
            total_size=num_pages * page,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(CB_Q_IN, 2 * D_t),  # Q block, held across KV loop (double-buffered)
        cb(CB_K_IN, 2 * D_t),  # K block, streamed
        cb(CB_V_IN, 2 * D_t),  # V block, streamed
        # Reduce scalers are bf16-packed by prepare_reduce_scaler — always bf16.
        cb(CB_SCALE, 1, ttnn.bfloat16),
        cb(CB_SCALER_MAX, 1, ttnn.bfloat16),
        cb(CB_SCALER_SUM, 1, ttnn.bfloat16),
        cb(CB_MAX, 2, accum_fmt),  # running max m_i (persists across KV loop)
        cb(CB_MAX_PREV, 2, accum_fmt),
        cb(CB_CORR, 2, accum_fmt),
        cb(CB_L, 2, accum_fmt),  # running sum l_i (persists)
        cb(CB_L_BLOCK, 2, accum_fmt),
        cb(CB_M_BLK, 2, accum_fmt),
        cb(CB_QK, 2, accum_fmt),  # score block S = Q.Kᵀ (precision lever)
        cb(CB_P, 2, accum_fmt),  # prob block P = exp(S - m)
        cb(CB_O_ACC, 2 * D_t, accum_fmt),  # running output O_i (persists)
        cb(CB_PV, 2 * D_t, accum_fmt),
        cb(CB_O_TMP, 2 * D_t, accum_fmt),
        cb(CB_OUT, 2 * D_t, out_fmt),
    ]
    if has_mask:
        cbs.append(cb(CB_MASK_IN, 2))  # mask block: input dtype (see note above)

    # --- Reader CT args ---
    reader_ct = [D_t, S_q_t, S_kv_t, h, mask_h, 1 if has_mask else 0, _f32_bits(scale)]
    reader_ct.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # --- Writer CT args ---
    writer_ct = [D_t, S_q_t, h]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # --- Compute CT args ---
    compute_ct = [D_t, S_kv_t, 1 if has_mask else 0]

    q_addr = query.buffer_address()
    k_addr = key.buffer_address()
    v_addr = value.buffer_address()
    mask_addr = attn_mask.buffer_address() if has_mask else 0
    out_addr = output_tensor.buffer_address()

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    start = 0
    for i, c in enumerate(cores):
        n = base + (1 if i < rem else 0)
        reader_rt[c.x][c.y] = [q_addr, k_addr, v_addr, mask_addr, start, n]
        writer_rt[c.x][c.y] = [out_addr, start, n]
        compute_rt[c.x][c.y] = [n]
        start += n

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        # Caller-controlled via compute_kernel_config (resolved above). Defaults
        # reproduce Phase 0: HiFi2 + fp32 DEST accumulation for the
        # online-softmax recurrence.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_approx_mode=math_approx_mode,
            dst_full_sync_en=dst_full_sync_en,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
