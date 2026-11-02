// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_gated_delta_rule.hpp"

#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

#include "device/fused_recurrent_gated_delta_rule_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/device.hpp"

using namespace tt::tt_metal;

namespace ttnn::transformer {

namespace {

ttnn::Tensor as_f32(const ttnn::Tensor& x) {
    return x.dtype() != DataType::FLOAT32 ? ttnn::typecast(x, DataType::FLOAT32) : x;
}

// [B,T,H,D] -> [B*H, T, D], fp32 TILE (head-major; permute on TILE via transpose engine).
ttnn::Tensor head_split(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t H, uint32_t D) {
    ttnn::Tensor t = as_f32(x);
    t = ttnn::permute(t, ttsl::SmallVector<int64_t>{0, 2, 1, 3});  // [B,H,T,D]
    return ttnn::reshape(t, ttnn::Shape({B * H, T, D}));
}

// Reshape through ROW_MAJOR (moves data across the tile boundary), then re-tilize.
ttnn::Tensor reshape_rm(const ttnn::Tensor& x, const ttnn::Shape& shape) {
    ttnn::Tensor t = ttnn::to_layout(x, Layout::ROW_MAJOR);
    t = ttnn::reshape(t, shape);
    return ttnn::to_layout(t, Layout::TILE);
}

// [B,T,H] -> [B*H*T, 1, 1] fp32 TILE: the per-(head, token) scalar pages the kernel broadcasts from
// (block index h*T + t, and bcast_scalar reads element [0,0] of each page).
//
// T == 1 fast path: the block order (h*1 + 0 = h) IS the element order of [B,1,H], so the entire
// layout change is ONE reshape across the tile boundary ([1,1,48] -> [48,1,1]) -- verified
// element-for-element against the row-major path in test_reshape_correct.py. The general form below
// also permutes and crosses the boundary twice (untilize + tilize); for the pair of scalars this
// serves, that measured 51.6 us per layer of pure layout cost, so it is kept only for T > 1, where
// the block order (h outer, t inner) really does differ from the element order (t outer).
ttnn::Tensor headvec_scalars(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t H) {
    ttnn::Tensor t = as_f32(x);
    if (T == 1) {
        return ttnn::reshape(t, ttnn::Shape({B * H, 1, 1}));
    }
    t = ttnn::permute(t, ttsl::SmallVector<int64_t>{0, 2, 1});  // [B,H,T]
    t = ttnn::reshape(t, ttnn::Shape({B * H, T}));
    return reshape_rm(t, ttnn::Shape({B * H * T, 1, 1}));
}

}  // namespace

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> fused_recurrent_gated_delta_rule(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& k_in,
    const ttnn::Tensor& v_in,
    const ttnn::Tensor& g_in,
    const ttnn::Tensor& beta_in,
    std::optional<float> scale_opt,
    const std::optional<ttnn::Tensor>& initial_state,
    const std::optional<ttnn::Tensor>& initial_state_block_idx,
    bool output_final_state,
    bool output_per_token_state,
    bool use_qk_l2norm,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool head_major_output,
    bool head_major_input,
    bool fused_gating) {
    TT_FATAL(!use_qk_l2norm, "fused_recurrent_gated_delta_rule: use_qk_l2norm not done here; L2-normalize q/k on host");

    auto* dev = q_in.device();
    const auto& qs = q_in.logical_shape();  // [B,T,H,K]; head_major_input: [H,1,K] (B = T = 1)
    const auto& vs = v_in.logical_shape();  // [B,T,HV,V]; head_major_input: [HV,1,V]
    TT_FATAL(
        !head_major_input || (qs.rank() == 3 && vs.rank() == 3 && qs[1] == 1 && vs[1] == 1),
        "head_major_input wants q/k [H,1,K] and v [HV,1,V] (one user, one token)");
    const uint32_t B = head_major_input ? 1 : qs[0];
    const uint32_t T = head_major_input ? 1 : qs[1];
    const uint32_t H = head_major_input ? qs[0] : qs[2];
    const uint32_t K = head_major_input ? qs[2] : qs[3];
    const uint32_t HV = head_major_input ? vs[0] : vs[2];
    const uint32_t V = head_major_input ? vs[2] : vs[3];
    TT_FATAL(HV % H == 0, "HV ({}) must be divisible by H ({})", HV, H);
    const uint32_t BH = B * HV;

    const float scale = scale_opt.has_value() ? *scale_opt : (1.0f / std::sqrt(static_cast<float>(K)));

    // Scale q (q is already L2-normed on host), then head-split + GQA-expand q,k from H to HV heads.
    ttnn::Tensor q = ttnn::multiply(as_f32(q_in), scale);
    ttnn::Tensor k = as_f32(k_in);
    // q/k keep their H query heads: the READER replicates head hv/G onto core hv, which costs one
    // integer divide per core instead of a repeat_interleave tensor. At decode that expand was two
    // untilize/concat/tilize/copy chains, ~11 us each.
    // head_major_input: the caller already holds q/k/v as [heads, T=1, D] (one head per tile-row), which is
    // what head_split produces, so the three permutes go.
    if (!head_major_input) {
        q = head_split(q, B, T, H, K);  // [B*H, T, K]  (H = HV/G query heads)
        k = head_split(k, B, T, H, K);
    }
    ttnn::Tensor v = head_major_input ? as_f32(v_in) : head_split(v_in, B, T, HV, V);  // [BH, T, V]

    // decay = exp(g); beta as-is (already sigmoid'd). exp is elementwise, so it commutes with the
    // layout change; doing it here keeps it on the compact [B,T,H] form (2 tiles at decode rather
    // than 48 padded pages).
    // fused_gating (QWEN36_1C_GDN_GB, decode): g_in is the raw a|b row [1,1,2HV] and beta_in the constants row
    // [1,1,2HV] (-exp(A_log) | dt_bias); the kernel computes decay and beta, so no exp and no scalar relayout here.
    TT_FATAL(!fused_gating || T == 1, "fused_recurrent_gated_delta_rule: fused_gating is decode-only (T == 1)");
    ttnn::Tensor decay = fused_gating ? as_f32(g_in) : ttnn::exp(as_f32(g_in));  // [B,T,H], unchanged layout
    ttnn::Tensor beta = as_f32(beta_in);                                          // [B,T,H]

    // Per-token layout: each (head, token) its own zero-padded tile-block.
    //   q/k [BH*T,1,K], v [BH*T,1,V], decay/beta [BH*T,1,1].
    // T==1 fast path: head_split already yields [BH,1,D] == [BH*T,1,D], so the q/k/v relayout is a
    // no-op — skip the RM round-trip (it is ~5x the cost of the permute; see test_headsplit_bench).
    if (T > 1) {
        const uint32_t BHQ = B * H;  // q/k hold H query heads, not HV
        q = reshape_rm(q, ttnn::Shape({BHQ * T, 1, K}));
        k = reshape_rm(k, ttnn::Shape({BHQ * T, 1, K}));
        v = reshape_rm(v, ttnn::Shape({BH * T, 1, V}));
    }
    // decay/beta must be per-scalar tiles [BH*T,1,1] so bcast_scalar reads element [0,0]. At T == 1
    // that is one reshape each (see headvec_scalars); the values are 48 floats, the cost is the
    // crossing, not the arithmetic.
    if (!fused_gating) {
        decay = headvec_scalars(decay, B, T, HV);
        beta = headvec_scalars(beta, B, T, HV);
    }

    // Initial state [B,HV,K,V] -> [BH,K,V] fp32 TILE. Always provide (zeros if absent) so the reader
    // always reads S (it takes the unconditional path; there is no in-kernel zeroing). Traced callers
    // pass a persistent state buffer (never absent); the zeros() fallback here is eager-only
    // (device-side fill, uncached), same caveat as chunk_gated_delta_rule.
    //
    // Ring mode: initial_state is the [T*BH,K,V] ring and also the state output; pass it through untouched, since a
    // reshape or typecast would break the in-place aliasing. The device op validates it.
    const bool ring = initial_state_block_idx.has_value();
    std::optional<ttnn::Tensor> s0 = initial_state;
    if (!ring) {
        if (initial_state.has_value()) {
            s0 = ttnn::reshape(as_f32(*initial_state), ttnn::Shape({BH, K, V}));
        } else {
            s0 = ttnn::zeros(
                ttnn::Shape({BH, K, V}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
        }
    }

    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    const bool want_state = output_final_state || output_per_token_state;
    auto res = ttnn::prim::fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        decay,
        beta,
        s0,
        initial_state_block_idx,
        T,
        want_state && !output_per_token_state,
        output_per_token_state,
        out_mem,
        kernel_cfg,
        fused_gating);
    const ttnn::Tensor& o_pt = res[0];  // [BH*T, 1, V]
    const ttnn::Tensor& st = res[1];    // [BH*T,K,V] per-token, else [BH,K,V]

    // o [BH*T,1,V] -> [B,HV,T,V] -> [B,T,HV,V]. head_major_output keeps the kernel's [BH*T,1,V] (one head per
    // tile-row): a per-head consumer (the gated norm) works on it as is, and at B == T == 1 its tile sequence
    // equals that of [1,1,HV*V], so the caller can flatten it with a view instead of untilize/permute/tilize.
    ttnn::Tensor o = o_pt;
    if (!head_major_output) {
        o = ttnn::to_layout(o_pt, Layout::ROW_MAJOR);
        o = ttnn::reshape(o, ttnn::Shape({B, HV, T, V}));
        o = ttnn::permute(o, ttsl::SmallVector<int64_t>{0, 2, 1, 3});  // [B,T,HV,V]
        o = ttnn::to_layout(o, Layout::TILE);
    }

    std::optional<ttnn::Tensor> state_opt;
    if (ring) {
        // res[1] is the caller's ring; return it as-is so the buffer stays aliased.
        state_opt = st;
    } else if (output_per_token_state) {
        // The writer lays per-token state out token-major, so the raw [BH*T,K,V] buffer already
        // reads as [T,B,HV,K,V]. At B==1 -- the speculative-decode case -- that IS the requested
        // [B,T,HV,K,V] element order, and K,V (the tiled dims) are untouched, so this is a pure
        // metadata reshape. It replaces an untilize + 5D permute + re-tilize of the whole state
        // tensor, which ran once per GDN layer per verify.
        if (B == 1) {
            state_opt = ttnn::reshape(st, ttnn::Shape({B, T, HV, K, V}));
        } else {
            ttnn::Tensor s = ttnn::to_layout(st, Layout::ROW_MAJOR);
            s = ttnn::reshape(s, ttnn::Shape({T, B, HV, K, V}));
            s = ttnn::permute(s, ttsl::SmallVector<int64_t>{1, 0, 2, 3, 4});  // [B,T,HV,K,V]
            state_opt = ttnn::to_layout(s, Layout::TILE);
        }
    } else if (output_final_state) {
        state_opt = ttnn::reshape(st, ttnn::Shape({B, HV, K, V}));
    }

    return {o, state_opt};
}

}  // namespace ttnn::transformer
