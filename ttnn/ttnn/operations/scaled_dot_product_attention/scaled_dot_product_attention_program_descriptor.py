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


def _largest_divisor_leq(n: int, cap: int) -> int:
    for c in range(min(cap, n), 0, -1):
        if n % c == 0:
            return c
    return 1


def _f32_bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", x))[0]


def create_program_descriptor(
    Q: ttnn.Tensor,
    K: ttnn.Tensor,
    V: ttnn.Tensor,
    attention_mask,
    output_tensor: ttnn.Tensor,
    *,
    scale: float,
) -> ttnn.ProgramDescriptor:
    device = Q.device()

    q_shape = list(Q.shape)
    k_shape = list(K.shape)
    B, H_q, S_q, D = q_shape
    H_kv, S_kv = k_shape[1], k_shape[2]

    DHt = D // TILE_DIM
    vDHt = DHt  # V head dim == D
    Sq_t = S_q // TILE_DIM
    Sk_t = S_kv // TILE_DIM

    B_q = _largest_divisor_leq(Sq_t, MAX_B_Q)
    B_kv = _largest_divisor_leq(Sk_t, MAX_B_KV)
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
        # reader-facing inputs: double-buffered for reader/compute pipelining
        cb(CB_Q, q_page, 2 * B_q * DHt, Q.dtype),
        cb(CB_K, k_page, 2 * B_kv * DHt, K.dtype),
        cb(CB_V, v_page, 2 * B_kv * vDHt, V.dtype),
        # persistent constants
        cb(CB_MAX_SCALER, bf16_tile, 1),
        cb(CB_SUM_SCALER, bf16_tile, 1),
        # per-iteration / persistent compute scratch (single-buffered: sequential)
        cb(CB_ALPHA, bf16_tile, B_q),
        cb(CB_L_RECIP, bf16_tile, B_q),
        cb(CB_M_BLK, bf16_tile, B_q),
        cb(CB_QK, bf16_tile, qk_tiles),
        cb(CB_P, bf16_tile, qk_tiles),
        cb(CB_O_BLK, bf16_tile, o_tiles),
        cb(CB_O_RUN, bf16_tile, o_tiles),
        cb(CB_M_PREV, bf16_tile, B_q),
        cb(CB_M_RUN, bf16_tile, B_q),
        cb(CB_L_RUN, bf16_tile, B_q),
        cb(CB_L_BLK, bf16_tile, B_q),
        # output: double-buffered for compute/writer pipelining
        cb(CB_OUT, out_page, 2 * o_tiles, output_tensor.dtype),
    ]
    if has_mask:
        cbs.append(cb(CB_MASK, mask_page, 2 * qk_tiles, attention_mask.dtype))

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
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        # bf16 inputs with Kt>1 K-accumulation: HiFi2 is the safe fidelity.
        # fp32_dest_acc is left off to avoid the known-bad HiFi4+fp32+bf16 combo
        # in the matmul-path SUM reduce (which hardcodes HiFi4).
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2),
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
