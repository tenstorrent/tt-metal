# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""compute_drain bench: one worker core of a2a_wkv_b2 (one head: Mt=5 tile rows of attn sticks, W resident,
tilize + matmul + local publish), with the reader's row landings paced at ROW_GAP so every variant sees the
same landing times. Variants differ only in the compute (and, for K-split, the reader's chunk order)."""
from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
CB_W, CB_RM, CB_TILED, CB_OUT, CB_INTERM = 0, 1, 2, 3, 4

# name -> (compute variant, KC_OTHER, KC_LAST, tiled rows)
VARIANTS = {
    "base": (0, 1, 1, 1),
    "lookahead": (1, 1, 1, 2),
    "ks1_helper": (2, 1, 1, 1),  # base math through the K-split code path (KC = 1)
    "ks2_last": (2, 1, 2, 1),
    "ks4_last": (2, 1, 4, 1),
    "ks2_all": (2, 2, 2, 1),
    "ks4_all": (2, 4, 4, 1),
    "chunkpush": (3, 1, 1, 1),
    "chunkpush_la": (4, 1, 1, 2),
    "fused": (5, 1, 1, 1),
}


def run(device, attn, w, variant, row_gap_cycles):
    cv, kc_other, kc_last, tiled_rows = VARIANTS[variant]
    _, _, T, L = attn.shape
    V = w.shape[-1]
    Mt, Kt, Nt = T // 32, L // 32, V // 32
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, V]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    core = ttnn.CoreCoord(0, 0)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    bf16_tile, bf8_tile = ttnn.tile_size(ttnn.bfloat16), ttnn.tile_size(ttnn.bfloat8_b)

    def cb(i, dt, page, n):
        return ttnn.CBDescriptor(
            total_size=page * n,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )

    cbs = [
        cb(CB_W, ttnn.bfloat8_b, bf8_tile, Kt * Nt),
        cb(CB_RM, ttnn.bfloat16, bf16_tile, 2 * Kt),
        cb(CB_TILED, ttnn.bfloat16, bf16_tile, tiled_rows * Kt),
        cb(CB_OUT, ttnn.bfloat8_b, bf8_tile, Mt * Nt),
        cb(CB_INTERM, ttnn.bfloat16, bf16_tile, Nt),
    ]
    rd_ct = [CB_W, CB_RM, Kt, Nt, Mt, L * 2, bf8_tile, kc_other, kc_last, row_gap_cycles]
    rd_ct += ttnn.TensorAccessorArgs(attn).get_compile_time_args()
    rd_ct += ttnn.TensorAccessorArgs(w).get_compile_time_args()
    rd_rt = ttnn.RuntimeArgs()
    rd_rt[0][0] = [attn.buffer_address(), w.buffer_address()]
    wr_ct = [CB_OUT, Mt, Nt, bf8_tile] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    wr_rt = ttnn.RuntimeArgs()
    wr_rt[0][0] = [out.buffer_address()]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "cd_reader.cpp"),
            core_ranges=crs,
            compile_time_args=rd_ct,
            runtime_args=rd_rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "cd_writer.cpp"),
            core_ranges=crs,
            compile_time_args=wr_ct,
            runtime_args=wr_rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "cd_compute.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_RM, CB_TILED, CB_W, CB_OUT, Kt, Nt, Mt, CB_INTERM, kc_other, kc_last],
            defines=[("CD_VARIANT", str(cv))],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
            ),
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([attn, w, out], prog)
