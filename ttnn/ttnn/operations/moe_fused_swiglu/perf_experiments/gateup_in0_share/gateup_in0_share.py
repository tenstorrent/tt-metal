# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED BAKE-OFF — moe_fused_swiglu gate/up matmul `in0`-unpack sharing.

Owner idea: cut the gate/up matmul's `in0` (resident `x`) unpack traffic by unpacking each `x` tile
ONCE for both the gate and up weight matrices, instead of once per matrix. The real op
(`kernels/moe_fused_swiglu_compute.cpp`, `compute_gateup` block) issues TWO back-to-back
`matmul_block` helper calls over the SAME resident `cb_x_tiles` — one for `W_gate`, one for `W_up` —
so every `x` tile is unpacked twice per M-block.

Single-core, sharded-L1-resident benchmark (no DRAM in the fast path, no reader/writer kernels) —
`ttnn.cb_descriptor_from_sharded_tensor` binds each operand directly as a CB, matching
`examples/matmul_output_subblock`'s pattern. This measures the COMPUTE lever cleanly, independent of
the real op's weight-DRAM coupling (see the report for how much of the isolated result can survive
that coupling).

Shapes mirror the real op's gate/up stage exactly:
  x:      bfloat8_b [m_eff*32, kr_pad*32]   (in0, resident, unpacked twice today)
  W_gate:  bfloat4_b [kr_pad*32, hn_pad*32] (in1 #1)
  W_up:    bfloat4_b [kr_pad*32, hn_pad*32] (in1 #2)
  gate/up out: bfloat8_b [m_eff*32, hn_pad*32] each (LastBlockTarget::Out, num_k_blocks == 1, no interm
  spill — matches moe_fused_swiglu_program_descriptor.py's CB_GATE_ACC / CB_UP_ACC format)

Variants:
  baseline      — two separate matmul_block calls (gate, then up) over the same resident cb_x. The
                  op's honest current approach.
  merged_1call  — ONE matmul_block call, in1_num_subblocks=2, over a CONCATENATED in1 CB
                  ([kr_pad, 2*hn_pad], W_gate columns then W_up columns) and a concatenated output CB.
                  Tests whether merging the CALL shares in0 across in1 sub-blocks (matmul_block_helpers
                  reads say NO: in0_index resets to the subblock offset at the top of EACH in1_subblock
                  iteration — this variant measures that hypothesis instead of concluding from the read).
  wide_subblock — single-matrix matmul (no gate/up split), N=24 tiles, sweeping out_subblock_w over
                  {2,3,4,6,8} (in1_num_subblocks = 24/out_subblock_w) at OUR dtypes (bfp8 in0 / bfp4
                  in1) and OUR kr. Total FMA count is identical across the sweep (same M/K/N) — this
                  isolates the marginal effect of a wider ct_dim (more in1 tiles amortizing one in0
                  unpack) that `examples/matmul_output_subblock` measured in bf16/bf16; re-measured
                  here in bfp8/bfp4 because the unpack:math ratio (and therefore the win) is
                  format-dependent.
  All variants support a SKIP_COMPUTE compile define (ablation: strips the ckernel::matmul_block LLK
  call — both unpack AND math — on every TRISC, keeping every wait/reserve/push/pop/pack-shell
  intact) to separate "unpack+math" cost from "CB sync + pack shell" cost.

PRECISION CONTRACT (fixed, never a lever): math_fidelity=LoFi, math_approx_mode=True,
fp32_dest_acc_en=False, dst_full_sync_en=False, bfp8_pack_precise=True — identical to
`moe_fused_swiglu.default_compute_kernel_config()`. Every variant runs under this SAME config.
"""

import ttnn

TILE = 32

# CB ids (buffer_index), matching the sharded-tensor / scratch CBs used by each variant.
CB_X = 0
CB_WG = 1
CB_WU = 2
CB_WGU = 3  # merged_1call: concatenated [W_gate | W_up]
CB_GATE = 16
CB_UP = 17
CB_GU = 18  # merged_1call: concatenated [gate_out | up_out]
CB_W_WIDE = 4  # wide_subblock: single weight matrix, N tiles wide
CB_ACC_WIDE = 19  # wide_subblock: single output matrix, N tiles wide

DEST_AUTO_LIMIT = 8  # fp16 DEST, half-sync (fp32_dest_acc_en=False, dst_full_sync_en=False) — fixed.

DEFAULT_CFG = dict(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    dst_full_sync_en=False,
    bfp8_pack_precise=True,
)


def compute_kernel_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = DEFAULT_CFG["math_fidelity"]
    cfg.math_approx_mode = DEFAULT_CFG["math_approx_mode"]
    cfg.fp32_dest_acc_en = DEFAULT_CFG["fp32_dest_acc_en"]
    cfg.dst_full_sync_en = DEFAULT_CFG["dst_full_sync_en"]
    cfg.bfp8_pack_precise = DEFAULT_CFG["bfp8_pack_precise"]
    return cfg


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(h_tiles, w_tiles):
    """The whole [h_tiles x w_tiles] tile matrix as one shard on a single core (tiles row-major)."""
    return ttnn.create_sharded_memory_config(
        shape=(h_tiles * TILE, w_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# =============================================================================
# Kernel sources
# =============================================================================

# Shared per-K-block FMA step count: the real per-row K extent (`kr`) may be smaller than the padded
# tile-row stride (`KR_PAD`) — mirrors moe_fused_swiglu_compute.cpp's KrSteps. Also used to narrow the
# gate/up matmul's REAL hidden width via last_in1_subblock_w_valid (the ragged ``hn < HN_PAD`` column).
_COMMON_PREAMBLE = r"""
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

namespace ckl = compute_kernel_lib;

struct KrSteps {
    uint32_t kr;
    ALWI uint32_t operator()(uint32_t, uint32_t) const { return kr; }
};
"""

# baseline: two separate matmul_block calls over the SAME resident cb_x — the op's honest current
# approach for the gate/up stage. Weights + x are resident (WaitAndRetainOnLastBlock on BOTH operands,
# same convention as examples/matmul_output_subblock) so `kernel_iters` repeats the identical work
# without any DRAM refill, giving a bigger, more reliable measurement window.
_BASELINE_KERNEL = (
    _COMMON_PREAMBLE
    + r"""
void kernel_main() {
    constexpr uint32_t M_EFF = get_compile_time_arg_val(0);
    constexpr uint32_t KR_PAD = get_compile_time_arg_val(1);
    constexpr uint32_t KR_REAL = get_compile_time_arg_val(2);
    constexpr uint32_t HN_PAD = get_compile_time_arg_val(3);
    constexpr uint32_t HN_REAL = get_compile_time_arg_val(4);   // 0 == full (inert), else ragged width
    constexpr uint32_t OUT_SUBBLOCK_H = get_compile_time_arg_val(5);  // pinned at 1 in the real op
    constexpr uint32_t KERNEL_ITERS = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = 0, cb_wg = 1, cb_wu = 2, cb_gate = 16, cb_up = 17;

    cb_reserve_back(cb_x, M_EFF * KR_PAD); cb_push_back(cb_x, M_EFF * KR_PAD);
    cb_reserve_back(cb_wg, KR_PAD * HN_PAD); cb_push_back(cb_wg, KR_PAD * HN_PAD);
    cb_reserve_back(cb_wu, KR_PAD * HN_PAD); cb_push_back(cb_wu, KR_PAD * HN_PAD);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_wg, cb_gate);

    CircularBuffer x_buf(cb_x), wg_buf(cb_wg), wu_buf(cb_wu), gate_buf(cb_gate), up_buf(cb_up);

    ckl::MatmulBlockShape shape =
        ckl::MatmulBlockShape::of(M_EFF / OUT_SUBBLOCK_H, 1, OUT_SUBBLOCK_H, HN_PAD, KR_PAD, 1);
    shape.last_in1_subblock_w_valid = HN_REAL;

    for (uint32_t iter = 0; iter < KERNEL_ITERS; ++iter) {
        ckl::matmul_block<
            /*transpose=*/false, /*packer_l1_acc=*/false, ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::SubblockMajor, ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndRetainOnLastBlock, ckl::InputPolicy::WaitAndRetainOnLastBlock,
            ckl::NoPostCompute, ckl::NoPreKBlock, ckl::NoPostKBlock, 0, KrSteps>(
            x_buf, wg_buf, gate_buf, gate_buf, shape, {}, {}, 0, 0, {}, KrSteps{KR_REAL});

        ckl::matmul_block<
            /*transpose=*/false, /*packer_l1_acc=*/false, ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::SubblockMajor, ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndRetainOnLastBlock, ckl::InputPolicy::WaitAndRetainOnLastBlock,
            ckl::NoPostCompute, ckl::NoPreKBlock, ckl::NoPostKBlock, 0, KrSteps>(
            x_buf, wu_buf, up_buf, up_buf, shape, {}, {}, 0, 0, {}, KrSteps{KR_REAL});

        if (iter + 1 < KERNEL_ITERS) {
            cb_wait_front(cb_gate, M_EFF * HN_PAD); cb_pop_front(cb_gate, M_EFF * HN_PAD);
            cb_wait_front(cb_up, M_EFF * HN_PAD); cb_pop_front(cb_up, M_EFF * HN_PAD);
        }
    }
    cb_pop_front(cb_x, M_EFF * KR_PAD);
    cb_pop_front(cb_wg, KR_PAD * HN_PAD);
    cb_pop_front(cb_wu, KR_PAD * HN_PAD);
}
"""
)

# merged_1call: ONE matmul_block call, in1_num_subblocks=2, over a CONCATENATED in1 CB
# ([W_gate | W_up], kr_pad x 2*hn_pad) and a concatenated output CB. Tests whether the helper's
# SubblockMajor walk (in0_subblock outer / in1_subblock inner) shares the in0 unpack across the two
# in1 sub-blocks it now sees — matmul_block_helpers.inl resets `in0_index` to the subblock offset at
# the top of EVERY in1_subblock iteration, so the prediction is NO SHARING (same total unpack bytes as
# baseline, module call/init overhead). Measured, not assumed.
_MERGED_KERNEL = (
    _COMMON_PREAMBLE
    + r"""
void kernel_main() {
    constexpr uint32_t M_EFF = get_compile_time_arg_val(0);
    constexpr uint32_t KR_PAD = get_compile_time_arg_val(1);
    constexpr uint32_t KR_REAL = get_compile_time_arg_val(2);
    constexpr uint32_t HN_PAD = get_compile_time_arg_val(3);
    constexpr uint32_t HN_REAL = get_compile_time_arg_val(4);
    constexpr uint32_t OUT_SUBBLOCK_H = get_compile_time_arg_val(5);
    constexpr uint32_t KERNEL_ITERS = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = 0, cb_wgu = 3, cb_gu = 18;

    cb_reserve_back(cb_x, M_EFF * KR_PAD); cb_push_back(cb_x, M_EFF * KR_PAD);
    cb_reserve_back(cb_wgu, KR_PAD * 2 * HN_PAD); cb_push_back(cb_wgu, KR_PAD * 2 * HN_PAD);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_wgu, cb_gu);

    CircularBuffer x_buf(cb_x), wgu_buf(cb_wgu), gu_buf(cb_gu);

    // Two in1 sub-blocks (gate cols, then up cols) in ONE call. last_in1_subblock_w_valid narrows
    // ONLY the last (up) sub-block, matching the ragged column contract if HN_REAL != 0.
    ckl::MatmulBlockShape shape =
        ckl::MatmulBlockShape::of(M_EFF / OUT_SUBBLOCK_H, 2, OUT_SUBBLOCK_H, HN_PAD, KR_PAD, 1);
    shape.last_in1_subblock_w_valid = HN_REAL;

    for (uint32_t iter = 0; iter < KERNEL_ITERS; ++iter) {
        ckl::matmul_block<
            /*transpose=*/false, /*packer_l1_acc=*/false, ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::SubblockMajor, ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndRetainOnLastBlock, ckl::InputPolicy::WaitAndRetainOnLastBlock,
            ckl::NoPostCompute, ckl::NoPreKBlock, ckl::NoPostKBlock, 0, KrSteps>(
            x_buf, wgu_buf, gu_buf, gu_buf, shape, {}, {}, 0, 0, {}, KrSteps{KR_REAL});

        if (iter + 1 < KERNEL_ITERS) {
            cb_wait_front(cb_gu, M_EFF * 2 * HN_PAD); cb_pop_front(cb_gu, M_EFF * 2 * HN_PAD);
        }
    }
    cb_pop_front(cb_x, M_EFF * KR_PAD);
    cb_pop_front(cb_wgu, KR_PAD * 2 * HN_PAD);
}
"""
)

# wide_subblock: single-matrix matmul C[M,N] = X[M,K] @ W[K,N], sweeping out_subblock_w. Total FMA
# count (M*K*N) is IDENTICAL across the sweep — only the subblocking changes. Isolates the marginal
# per-FMA cost of a wider ct_dim (matmul_output_subblock's reuse-A mechanism) at OUR dtypes (bfp8
# in0 / bfp4 in1) and OUR kr, decoupled from the two-matrix in0-sharing question.
_WIDE_KERNEL = (
    _COMMON_PREAMBLE
    + r"""
void kernel_main() {
    constexpr uint32_t M_EFF = get_compile_time_arg_val(0);
    constexpr uint32_t KR_PAD = get_compile_time_arg_val(1);
    constexpr uint32_t KR_REAL = get_compile_time_arg_val(2);
    constexpr uint32_t N_TOTAL = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_SUBBLOCK_W = get_compile_time_arg_val(4);
    constexpr uint32_t OUT_SUBBLOCK_H = get_compile_time_arg_val(5);
    constexpr uint32_t KERNEL_ITERS = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = 0, cb_w = 4, cb_acc = 19;
    constexpr uint32_t IN1_SUBBLOCKS = N_TOTAL / OUT_SUBBLOCK_W;

    cb_reserve_back(cb_x, M_EFF * KR_PAD); cb_push_back(cb_x, M_EFF * KR_PAD);
    cb_reserve_back(cb_w, KR_PAD * N_TOTAL); cb_push_back(cb_w, KR_PAD * N_TOTAL);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_acc);

    CircularBuffer x_buf(cb_x), w_buf(cb_w), acc_buf(cb_acc);

    ckl::MatmulBlockShape shape = ckl::MatmulBlockShape::of(
        M_EFF / OUT_SUBBLOCK_H, IN1_SUBBLOCKS, OUT_SUBBLOCK_H, OUT_SUBBLOCK_W, KR_PAD, 1);

    for (uint32_t iter = 0; iter < KERNEL_ITERS; ++iter) {
        ckl::matmul_block<
            /*transpose=*/false, /*packer_l1_acc=*/false, ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::SubblockMajor, ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndRetainOnLastBlock, ckl::InputPolicy::WaitAndRetainOnLastBlock,
            ckl::NoPostCompute, ckl::NoPreKBlock, ckl::NoPostKBlock, 0, KrSteps>(
            x_buf, w_buf, acc_buf, acc_buf, shape, {}, {}, 0, 0, {}, KrSteps{KR_REAL});

        if (iter + 1 < KERNEL_ITERS) {
            cb_wait_front(cb_acc, M_EFF * N_TOTAL); cb_pop_front(cb_acc, M_EFF * N_TOTAL);
        }
    }
    cb_pop_front(cb_x, M_EFF * KR_PAD);
    cb_pop_front(cb_w, KR_PAD * N_TOTAL);
}
"""
)


# =============================================================================
# Program descriptors
# =============================================================================


def create_program_descriptor_baseline(
    x, wg, wu, gate_out, up_out, *, kr_pad, kr_real, hn_pad, hn_real, m_eff, kernel_iters, skip_compute=False
):
    out_subblock_h = 1  # pinned in the real op — the gate/up output must stay m-major (op_design §1.4)
    compute = ttnn.KernelDescriptor(
        kernel_source=_BASELINE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[m_eff, kr_pad, kr_real, hn_pad, hn_real, out_subblock_h, kernel_iters],
        defines=[("SKIP_COMPUTE", "1")] if skip_compute else [],
        config=compute_kernel_config(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_WG, wg),
        ttnn.cb_descriptor_from_sharded_tensor(CB_WU, wu),
        ttnn.cb_descriptor_from_sharded_tensor(CB_GATE, gate_out),
        ttnn.cb_descriptor_from_sharded_tensor(CB_UP, up_out),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def create_program_descriptor_merged(
    x, wgu, gu_out, *, kr_pad, kr_real, hn_pad, hn_real, m_eff, kernel_iters, skip_compute=False
):
    out_subblock_h = 1
    compute = ttnn.KernelDescriptor(
        kernel_source=_MERGED_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[m_eff, kr_pad, kr_real, hn_pad, hn_real, out_subblock_h, kernel_iters],
        defines=[("SKIP_COMPUTE", "1")] if skip_compute else [],
        config=compute_kernel_config(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_WGU, wgu),
        ttnn.cb_descriptor_from_sharded_tensor(CB_GU, gu_out),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def create_program_descriptor_wide(x, w, acc_out, *, kr_pad, kr_real, n_total, out_subblock_w, m_eff, kernel_iters):
    out_subblock_h = 1
    if n_total % out_subblock_w != 0:
        raise ValueError(f"n_total {n_total} must be divisible by out_subblock_w {out_subblock_w}")
    if out_subblock_h * out_subblock_w > DEST_AUTO_LIMIT:
        raise ValueError(f"subblock {out_subblock_h}x{out_subblock_w} exceeds DEST budget {DEST_AUTO_LIMIT}")
    compute = ttnn.KernelDescriptor(
        kernel_source=_WIDE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[m_eff, kr_pad, kr_real, n_total, out_subblock_w, out_subblock_h, kernel_iters],
        config=compute_kernel_config(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_W_WIDE, w),
        ttnn.cb_descriptor_from_sharded_tensor(CB_ACC_WIDE, acc_out),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
