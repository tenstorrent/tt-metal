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
) -> ttnn.ProgramDescriptor:
    device = query.device()

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

    tile_size = ttnn.tile_size(query.dtype)

    # --- Circular buffers ---
    def cb(index, num_pages, fmt=query.dtype):
        return ttnn.CBDescriptor(
            total_size=num_pages * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=tile_size)],
        )

    cbs = [
        cb(CB_Q_IN, 2 * D_t),  # Q block, held across KV loop (double-buffered)
        cb(CB_K_IN, 2 * D_t),  # K block, streamed
        cb(CB_V_IN, 2 * D_t),  # V block, streamed
        cb(CB_SCALE, 1, ttnn.bfloat16),
        cb(CB_SCALER_MAX, 1, ttnn.bfloat16),
        cb(CB_SCALER_SUM, 1, ttnn.bfloat16),
        cb(CB_MAX, 2),  # >=2: in-place BinaryMax reserves while 1 held
        cb(CB_MAX_PREV, 2),
        cb(CB_CORR, 2),
        cb(CB_L, 2),
        cb(CB_L_BLOCK, 2),
        cb(CB_M_BLK, 2),
        cb(CB_QK, 2),
        cb(CB_P, 2),
        cb(CB_O_ACC, 2 * D_t),
        cb(CB_PV, 2 * D_t),
        cb(CB_O_TMP, 2 * D_t),
        cb(CB_OUT, 2 * D_t, output_tensor.dtype),
    ]
    if has_mask:
        cbs.append(cb(CB_MASK_IN, 2))

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
        # fp32 DEST accumulation for the online-softmax recurrence; HiFi2 (NOT
        # HiFi4, which is known-bad with bf16 + fp32_dest_acc on Wormhole B0).
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
