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
    t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1, 3});  // [B,H,T,D]
    return ttnn::reshape(t, ttnn::Shape({B * H, T, D}));
}

// [B,T,H] -> [B*H, T], fp32 TILE.
ttnn::Tensor headvec_split(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t H) {
    ttnn::Tensor t = as_f32(x);
    t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1});  // [B,H,T]
    return ttnn::reshape(t, ttnn::Shape({B * H, T}));
}

// Reshape through ROW_MAJOR (moves data across the tile boundary), then re-tilize.
ttnn::Tensor reshape_rm(const ttnn::Tensor& x, const ttnn::Shape& shape) {
    ttnn::Tensor t = ttnn::to_layout(x, Layout::ROW_MAJOR);
    t = ttnn::reshape(t, shape);
    return ttnn::to_layout(t, Layout::TILE);
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
    bool output_final_state,
    bool output_per_token_state,
    bool use_qk_l2norm,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(!use_qk_l2norm, "fused_recurrent_gated_delta_rule: use_qk_l2norm not done here; L2-normalize q/k on host");

    auto* dev = q_in.device();
    const auto& qs = q_in.logical_shape();  // [B,T,H,K]
    const auto& vs = v_in.logical_shape();  // [B,T,HV,V]
    const uint32_t B = qs[0];
    const uint32_t T = qs[1];
    const uint32_t H = qs[2];
    const uint32_t K = qs[3];
    const uint32_t HV = vs[2];
    const uint32_t V = vs[3];
    TT_FATAL(HV % H == 0, "HV ({}) must be divisible by H ({})", HV, H);
    const uint32_t G = HV / H;
    const uint32_t BH = B * HV;

    const float scale = scale_opt.has_value() ? *scale_opt : (1.0f / std::sqrt(static_cast<float>(K)));

    // Scale q (q is already L2-normed on host), then head-split + GQA-expand q,k from H to HV heads.
    ttnn::Tensor q = ttnn::multiply(as_f32(q_in), scale);
    ttnn::Tensor k = as_f32(k_in);
    q = head_split(q, B, T, H, K);  // [B*H, T, K]
    k = head_split(k, B, T, H, K);
    if (G > 1) {
        q = ttnn::repeat_interleave(q, G, 0);  // [BH, T, K]
        k = ttnn::repeat_interleave(k, G, 0);
    }
    ttnn::Tensor v = head_split(v_in, B, T, HV, V);  // [BH, T, V]

    // decay = exp(g); beta as-is (already sigmoid'd).
    ttnn::Tensor decay = ttnn::exp(headvec_split(g_in, B, T, HV));  // [BH, T]
    ttnn::Tensor beta = headvec_split(beta_in, B, T, HV);           // [BH, T]

    // Per-token layout: each (head, token) its own zero-padded tile-block.
    //   q/k [BH*T,1,K], v [BH*T,1,V], decay/beta [BH*T,1,1].
    // T==1 fast path: head_split already yields [BH,1,D] == [BH*T,1,D], so the q/k/v relayout is a
    // no-op — skip the RM round-trip (it is ~5x the cost of the permute; see test_headsplit_bench).
    if (T > 1) {
        q = reshape_rm(q, ttnn::Shape({BH * T, 1, K}));
        k = reshape_rm(k, ttnn::Shape({BH * T, 1, K}));
        v = reshape_rm(v, ttnn::Shape({BH * T, 1, V}));
    }
    // decay/beta come from headvec_split as [BH,T] (BH packed in tile-rows); they always need the
    // relayout to per-scalar tiles [BH*T,1,1] so bcast_scalar reads element [0,0]. They are tiny.
    decay = reshape_rm(decay, ttnn::Shape({BH * T, 1, 1}));
    beta = reshape_rm(beta, ttnn::Shape({BH * T, 1, 1}));

    // Initial state [B,HV,K,V] -> [BH,K,V] fp32 TILE. Always provide (zeros if absent) so the reader
    // always reads S (it takes the unconditional path; there is no in-kernel zeroing). Traced callers
    // pass a persistent state buffer (never absent); the zeros() fallback here is eager-only
    // (device-side fill, uncached), same caveat as chunk_gated_delta_rule.
    std::optional<ttnn::Tensor> s0;
    if (initial_state.has_value()) {
        s0 = ttnn::reshape(as_f32(*initial_state), ttnn::Shape({BH, K, V}));
    } else {
        s0 = ttnn::zeros(
            ttnn::Shape({BH, K, V}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
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
        T,
        want_state && !output_per_token_state,
        output_per_token_state,
        out_mem,
        kernel_cfg);
    ttnn::Tensor o_pt = res[0];  // [BH*T, 1, V]
    ttnn::Tensor st = res[1];    // [BH*T,K,V] per-token, else [BH,K,V]

    // o [BH*T,1,V] -> [B,HV,T,V] -> [B,T,HV,V].
    ttnn::Tensor o = ttnn::to_layout(o_pt, Layout::ROW_MAJOR);
    o = ttnn::reshape(o, ttnn::Shape({B, HV, T, V}));
    o = ttnn::permute(o, ttnn::SmallVector<int64_t>{0, 2, 1, 3});  // [B,T,HV,V]
    o = ttnn::to_layout(o, Layout::TILE);

    std::optional<ttnn::Tensor> state_opt;
    if (output_per_token_state) {
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
            s = ttnn::permute(s, ttnn::SmallVector<int64_t>{1, 0, 2, 3, 4});  // [B,T,HV,K,V]
            state_opt = ttnn::to_layout(s, Layout::TILE);
        }
    } else if (output_final_state) {
        state_opt = ttnn::reshape(st, ttnn::Shape({B, HV, K, V}));
    }

    return {o, state_opt};
}

}  // namespace ttnn::transformer
