// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Host side of fused_recurrent_gated_delta_rule: the v2 contract (shape rules, combination rules, the staged
// rejections) in front of the program today's kernel serves. The order of the checks is fixed so a given bad call
// always fails on the same message: structural rules first, then the combination rules of the requirements' A-5,
// then what this build cannot serve, then the device operation's own checks.

#include "fused_recurrent_gated_delta_rule.hpp"

#include <cmath>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include <fmt/format.h>

#include "device/fused_recurrent_gated_delta_rule_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/device.hpp"

using namespace tt::tt_metal;

namespace ttnn::transformer {

namespace {

constexpr const char* kOp = "fused_recurrent_gated_delta_rule";

ttnn::Tensor as_f32(const ttnn::Tensor& x) {
    return x.dtype() != DataType::FLOAT32 ? ttnn::typecast(x, DataType::FLOAT32) : x;
}

// [B,T,H,D] -> [B*H, T, D], fp32 TILE (head-major; permute on TILE via transpose engine).
ttnn::Tensor head_split(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t H, uint32_t D) {
    ttnn::Tensor t = as_f32(x);
    t = ttnn::permute(t, ttsl::SmallVector<int64_t>{0, 2, 1, 3});  // [B,H,T,D]
    return ttnn::reshape(t, ttnn::Shape({B * H, T, D}));
}

// [B,T,H] -> [B*H, T], fp32 TILE.
ttnn::Tensor headvec_split(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t H) {
    ttnn::Tensor t = as_f32(x);
    t = ttnn::permute(t, ttsl::SmallVector<int64_t>{0, 2, 1});  // [B,H,T]
    return ttnn::reshape(t, ttnn::Shape({B * H, T}));
}

// Reshape through ROW_MAJOR (moves data across the tile boundary), then re-tilize.
ttnn::Tensor reshape_rm(const ttnn::Tensor& x, const ttnn::Shape& shape) {
    ttnn::Tensor t = ttnn::to_layout(x, Layout::ROW_MAJOR);
    t = ttnn::reshape(t, shape);
    return ttnn::to_layout(t, Layout::TILE);
}

// use_qk_l2norm_in_kernel: FLA's norm x / sqrt(sum_K x^2 + eps) in fp32 with eps inside the root, the form the v2
// requirements fix as the reference, as host ops in front of the program until stage 4 moves it into the reader.
// Eight device ops when enabled, none when not, so the default call's op count does not move. Deliberately not the
// model's rms_norm-based approximation: that difference is the documented numerics change of the migration.
ttnn::Tensor l2_norm_rows(const ttnn::Tensor& x_in, const DeviceComputeKernelConfig& cfg) {
    const ttnn::Tensor x = as_f32(x_in);
    const ttnn::Tensor sum_sq = ttnn::sum(ttnn::multiply(x, x), -1, /*keepdim=*/true, std::nullopt, cfg);
    const ttnn::Tensor inv = ttnn::rsqrt(ttnn::add(sum_sq, 1e-6f));
    return ttnn::multiply(x, inv);
}

// Stage-gated rejection: the signature accepts the argument, this build cannot serve it. One message shape for
// every such case, so a caller can read which stage of the v2 plan brings the feature.
void not_in_this_build(const std::string& feature, const char* stage, const char* what) {
    TT_FATAL(
        false,
        "fused_recurrent_gated_delta_rule: {} is not implemented in this build; the v2 plan brings it in stage {} ({})",
        feature,
        stage,
        what);
}

// The dimensions a call names, read off q, v and g.
struct Dims {
    uint32_t B, T, H, K, HV, V;
    bool kda;  // rank-4 g: a per-key decay vector
};

bool is_floating_input(const ttnn::Tensor& t) {
    return t.dtype() == DataType::BFLOAT16 || t.dtype() == DataType::FLOAT32;
}

void require_on_device(const ttnn::Tensor& t, const char* name, const ttnn::Tensor& q) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr,
        "{}: {} must be allocated on device",
        kOp,
        name);
    TT_FATAL(t.device() == q.device(), "{}: {} must be on the same device as q", kOp, name);
}

// Structural rules R1-R4: ranks, consistent dimensions, dtypes, head dimensions, the token cap.
Dims check_shapes(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta) {
    const auto& qs = q.logical_shape();
    const auto& ks = k.logical_shape();
    const auto& vs = v.logical_shape();
    const auto& gs = g.logical_shape();
    const auto& bs = beta.logical_shape();
    TT_FATAL(qs.rank() == 4, "{}: q must be [B,T,H,K] (rank 4), got rank {}", kOp, qs.rank());
    TT_FATAL(ks == qs, "{}: k must have q's shape [B,T,H,K], got {} vs {}", kOp, ks, qs);
    Dims d{};
    d.B = qs[0];
    d.T = qs[1];
    d.H = qs[2];
    d.K = qs[3];
    TT_FATAL(
        vs.rank() == 4 && vs[0] == qs[0] && vs[1] == qs[1],
        "{}: v must be [B,T,HV,V] (rank 4) with q's B and T, got {}",
        kOp,
        vs);
    d.HV = vs[2];
    d.V = vs[3];
    d.kda = gs.rank() == 4;
    const bool g_ok = (gs.rank() == 3 && gs[0] == d.B && gs[1] == d.T && gs[2] == d.HV) ||
                      (gs.rank() == 4 && gs[0] == d.B && gs[1] == d.T && gs[2] == d.HV && gs[3] == d.K);
    TT_FATAL(g_ok, "{}: g must be [B,T,HV] (GDN) or [B,T,HV,K] (KDA), got {}", kOp, gs);
    TT_FATAL(
        bs.rank() == 3 && bs[0] == d.B && bs[1] == d.T && bs[2] == d.HV, "{}: beta must be [B,T,HV], got {}", kOp, bs);
    TT_FATAL(
        d.B >= 1 && d.T >= 1 && d.H >= 1, "{}: B, T and H must be at least 1, got B {} T {} H {}", kOp, d.B, d.T, d.H);
    TT_FATAL(d.HV % d.H == 0, "{}: HV ({}) must be a multiple of H ({})", kOp, d.HV, d.H);
    TT_FATAL(
        d.K >= 32 && d.K % 32 == 0 && d.V >= 32 && d.V % 32 == 0,
        "{}: K and V must be non-zero multiples of 32, got K {} and V {}",
        kOp,
        d.K,
        d.V);
    TT_FATAL(
        d.T <= kFusedRecurrentGdnMaxTokensPerCall,
        "{}: T ({}) exceeds the maximum of {} tokens per call",
        kOp,
        d.T,
        kFusedRecurrentGdnMaxTokensPerCall);
    for (const auto& [t, name] : {std::pair{&q, "q"}, {&k, "k"}, {&v, "v"}, {&g, "g"}, {&beta, "beta"}}) {
        require_on_device(*t, name, q);
        TT_FATAL(is_floating_input(*t), "{}: {} must be bfloat16 or float32, got {}", kOp, name, t->dtype());
    }
    return d;
}

// Combination rules R5-R15 of the requirements' A-5: every rule is checked whether or not this build serves the
// feature, so the messages are stable across the stages that implement the features.
void check_rules(
    const Dims& d,
    const std::optional<ttnn::Tensor>& initial_state,
    const std::optional<ttnn::Tensor>& state_indices,
    const std::optional<ttnn::Tensor>& num_accepted_tokens,
    bool inplace_final_state,
    bool output_per_token_state,
    bool use_gate_in_kernel,
    const std::optional<ttnn::Tensor>& A_log,
    const std::optional<ttnn::Tensor>& dt_bias,
    std::optional<float> lower_bound,
    bool use_beta_sigmoid_in_kernel,
    bool allow_neg_eigval,
    const std::optional<ttnn::Tensor>& cu_seqlens,
    const std::optional<ttnn::Tensor>& output_tensor,
    const std::optional<ttnn::Tensor>& initial_state_block_idx,
    const ttnn::Tensor& q) {
    const bool ring = initial_state_block_idx.has_value();
    const bool pooled = state_indices.has_value();

    // R5 varlen
    std::optional<uint32_t> num_seqs;
    if (cu_seqlens.has_value()) {
        TT_FATAL(d.B == 1, "{}: cu_seqlens requires B == 1, got B {}", kOp, d.B);
        const auto& cs = cu_seqlens->logical_shape();
        TT_FATAL(
            cu_seqlens->dtype() == DataType::INT32 && cs.rank() == 1 && cs[0] >= 2,
            "{}: cu_seqlens must be int32 [N+1] with N >= 1, got {} {}",
            kOp,
            cu_seqlens->dtype(),
            cs);
        num_seqs = static_cast<uint32_t>(cs[0]) - 1;
    }
    // R6, R7, R8 gates
    TT_FATAL(
        !allow_neg_eigval || use_beta_sigmoid_in_kernel,
        "{}: allow_neg_eigval requires use_beta_sigmoid_in_kernel",
        kOp);
    TT_FATAL(!lower_bound.has_value() || d.kda, "{}: lower_bound applies to the KDA gate (rank-4 g) only", kOp);
    if (use_gate_in_kernel) {
        TT_FATAL(A_log.has_value() && dt_bias.has_value(), "{}: use_gate_in_kernel requires A_log and dt_bias", kOp);
        const auto& as = A_log->logical_shape();
        TT_FATAL(as.rank() == 1 && as[0] == d.HV, "{}: A_log must be [HV], got {}", kOp, as);
        const auto& ds = dt_bias->logical_shape();
        const uint32_t want = d.kda ? d.HV * d.K : d.HV;
        TT_FATAL(
            ds.rank() == 1 && ds[0] == want, "{}: dt_bias must be [HV] for GDN and [HV*K] for KDA, got {}", kOp, ds);
    } else {
        TT_FATAL(
            !A_log.has_value() && !dt_bias.has_value(),
            "{}: A_log and dt_bias apply only with use_gate_in_kernel",
            kOp);
    }
    // R9-R12 the pool
    TT_FATAL(
        !inplace_final_state || (initial_state.has_value() && pooled),
        "{}: inplace_final_state requires initial_state (the pool) and state_indices",
        kOp);
    if (num_accepted_tokens.has_value()) {
        TT_FATAL(
            pooled && state_indices->logical_shape().rank() == 2,
            "{}: num_accepted_tokens requires 2-D state_indices",
            kOp);
        const auto& ns = num_accepted_tokens->logical_shape();
        TT_FATAL(
            num_accepted_tokens->dtype() == DataType::INT32 && ns.rank() == 1 && ns[0] == d.B,
            "{}: num_accepted_tokens must be int32 [B], got {} {}",
            kOp,
            num_accepted_tokens->dtype(),
            ns);
    }
    if (pooled) {
        const auto& is = state_indices->logical_shape();
        const bool idx_ok = state_indices->dtype() == DataType::INT32 &&
                            ((is.rank() == 1 && is[0] == d.B) || (is.rank() == 2 && is[0] == d.B && is[1] == d.T));
        TT_FATAL(idx_ok, "{}: state_indices must be int32 [B] or [B, T], got {} {}", kOp, state_indices->dtype(), is);
        TT_FATAL(!ring, "{}: initial_state_block_idx (ring mode) and state_indices are exclusive", kOp);
        TT_FATAL(initial_state.has_value(), "{}: state_indices requires initial_state (the pool)", kOp);
        const auto& ps = initial_state->logical_shape();
        TT_FATAL(
            ps.rank() == 4 && ps[0] >= 1 && ps[1] == d.HV && ps[2] == d.K && ps[3] == d.V,
            "{}: with state_indices, initial_state is the pool [slots, HV, K, V], got {}",
            kOp,
            ps);
    } else if (initial_state.has_value() && !ring) {
        // R13 (R5 for varlen): the direct state
        const auto& ss = initial_state->logical_shape();
        const uint32_t rows = num_seqs.value_or(d.B);
        if (num_seqs.has_value()) {
            TT_FATAL(
                ss.rank() == 4 && ss[0] == rows && ss[1] == d.HV && ss[2] == d.K && ss[3] == d.V,
                "{}: initial_state must have one entry per sequence of cu_seqlens (N = {}), got {}",
                kOp,
                rows,
                ss);
        } else {
            TT_FATAL(
                ss.rank() == 4 && ss[0] == d.B && ss[1] == d.HV && ss[2] == d.K && ss[3] == d.V,
                "{}: initial_state must be [B, HV, K, V], got {}",
                kOp,
                ss);
        }
    }
    if (initial_state.has_value()) {
        require_on_device(*initial_state, "initial_state", q);
    }
    // R14 the preallocated output
    if (output_tensor.has_value()) {
        const auto& os = output_tensor->logical_shape();
        const bool ok = os.rank() == 4 && os[0] == d.B && os[1] == d.T && os[2] == d.HV && os[3] == d.V &&
                        output_tensor->dtype() == q.dtype() && output_tensor->layout() == Layout::TILE &&
                        output_tensor->storage_type() == StorageType::DEVICE && output_tensor->device() == q.device();
        TT_FATAL(
            ok, "{}: output_tensor must be [B, T, HV, V] in the input dtype, TILE, on q's device, got {}", kOp, os);
    }
    // R15 ring mode (the device operation checks the ring's shape and the index vector)
    TT_FATAL(!ring || output_per_token_state, "{}: initial_state_block_idx requires output_per_token_state", kOp);
}

// What this build cannot serve, by the stage of the v2 plan that brings it. Checked after every rule above.
void reject_unimplemented(
    const Dims& d,
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const std::optional<ttnn::Tensor>& state_indices,
    const std::optional<ttnn::Tensor>& num_accepted_tokens,
    bool inplace_final_state,
    bool use_gate_in_kernel,
    bool use_beta_sigmoid_in_kernel,
    const std::optional<ttnn::Tensor>& cu_seqlens,
    bool state_v_first,
    const std::optional<ttnn::Tensor>& output_tensor,
    uint32_t num_slices,
    uint32_t units_per_core) {
    auto* dev = q.device();
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    if (d.B * d.HV > grid.x * grid.y) {
        not_in_this_build(
            fmt::format("B*HV = {} above the {} compute cores (one core per head)", d.B * d.HV, grid.x * grid.y),
            "2",
            "V-split units and units looped per core, chosen by a cost model");
    }
    if (num_slices != 1 || units_per_core != 1) {
        not_in_this_build(
            "a program_config geometry other than one whole head per core",
            "2",
            "V-split units and units looped per core, chosen by a cost model");
    }
    if (state_indices.has_value() || num_accepted_tokens.has_value() || inplace_final_state) {
        not_in_this_build(
            "state_indices, num_accepted_tokens and inplace_final_state (the indexed state pool)",
            "3a",
            "an indexed, in-place state pool with per-token slots and accepted-token rollback");
    }
    if (output_tensor.has_value()) {
        not_in_this_build("output_tensor", "4", "the kernel rewrite writes o token-major, in place when asked");
    }
    for (const auto* t : {&q, &k, &v, &g, &beta}) {
        if (t->layout() != Layout::TILE) {
            not_in_this_build("ROW_MAJOR inputs", "4", "the kernel rewrite reads token-major inputs in either layout");
        }
    }
    if (d.kda || use_gate_in_kernel || use_beta_sigmoid_in_kernel) {
        not_in_this_build("KDA (rank-4 g) and the in-kernel gate options", "5", "KDA and in-kernel gating");
    }
    if (state_v_first) {
        not_in_this_build("state_v_first", "8", "the V-first pool layout decision");
    }
    TT_FATAL(
        !cu_seqlens.has_value(),
        "{}: cu_seqlens (varlen) is not scheduled in the v2 plan; run one call per sequence instead",
        kOp);
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
    const std::optional<ttnn::Tensor>& state_indices,
    const std::optional<ttnn::Tensor>& num_accepted_tokens,
    bool inplace_final_state,
    bool output_final_state,
    bool output_per_token_state,
    bool use_qk_l2norm_in_kernel,
    bool use_gate_in_kernel,
    const std::optional<ttnn::Tensor>& A_log,
    const std::optional<ttnn::Tensor>& dt_bias,
    std::optional<float> lower_bound,
    bool use_beta_sigmoid_in_kernel,
    bool allow_neg_eigval,
    const std::optional<ttnn::Tensor>& cu_seqlens,
    bool state_v_first,
    const std::optional<ttnn::Tensor>& output_tensor,
    const std::optional<ttnn::Tensor>& initial_state_block_idx,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<FusedRecurrentGdnProgramConfig>& program_config) {
    // Structural rules, then the combination rules of A-5 (every rule, whether or not this build serves the
    // feature), then what this build cannot serve.
    const Dims d = check_shapes(q_in, k_in, v_in, g_in, beta_in);
    const uint32_t num_slices = program_config.has_value() ? program_config->num_slices.value_or(1u) : 1u;
    const uint32_t units_per_core = program_config.has_value() ? program_config->units_per_core.value_or(1u) : 1u;
    TT_FATAL(
        num_slices == 1 || num_slices == 2 || num_slices == 4,
        "{}: program_config.num_slices must be 1, 2 or 4, got {}",
        kOp,
        num_slices);
    TT_FATAL(units_per_core >= 1, "{}: program_config.units_per_core must be at least 1, got {}", kOp, units_per_core);
    check_rules(
        d,
        initial_state,
        state_indices,
        num_accepted_tokens,
        inplace_final_state,
        output_per_token_state,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        lower_bound,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        cu_seqlens,
        output_tensor,
        initial_state_block_idx,
        q_in);
    reject_unimplemented(
        d,
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        state_indices,
        num_accepted_tokens,
        inplace_final_state,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        cu_seqlens,
        state_v_first,
        output_tensor,
        num_slices,
        units_per_core);

    auto* dev = q_in.device();
    const uint32_t B = d.B;
    const uint32_t T = d.T;
    const uint32_t H = d.H;
    const uint32_t K = d.K;
    const uint32_t HV = d.HV;
    const uint32_t V = d.V;
    const uint32_t G = HV / H;
    const uint32_t BH = B * HV;

    const float scale = scale_opt.has_value() ? *scale_opt : (1.0f / std::sqrt(static_cast<float>(K)));
    const auto kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    // The optional norm, then the scale (in that order), then head-split + GQA-expand q,k from H to HV heads.
    ttnn::Tensor q = use_qk_l2norm_in_kernel ? l2_norm_rows(q_in, kernel_cfg) : as_f32(q_in);
    q = ttnn::multiply(q, scale);
    ttnn::Tensor k = use_qk_l2norm_in_kernel ? l2_norm_rows(k_in, kernel_cfg) : as_f32(k_in);
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

    const bool want_state = output_final_state || output_per_token_state;
    auto res = ttnn::prim::fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        decay,
        beta,
        s0,
        initial_state_block_idx,
        B,
        HV,
        H,
        T,
        num_slices,
        units_per_core,
        want_state && !output_per_token_state,
        output_per_token_state,
        out_mem,
        kernel_cfg);
    const ttnn::Tensor& o_pt = res[0];  // [BH*T, 1, V]
    const ttnn::Tensor& st = res[1];    // [BH*T,K,V] per-token, else [BH,K,V]

    // o [BH*T,1,V] -> [B,HV,T,V] -> [B,T,HV,V].
    ttnn::Tensor o = ttnn::to_layout(o_pt, Layout::ROW_MAJOR);
    o = ttnn::reshape(o, ttnn::Shape({B, HV, T, V}));
    o = ttnn::permute(o, ttsl::SmallVector<int64_t>{0, 2, 1, 3});  // [B,T,HV,V]
    o = ttnn::to_layout(o, Layout::TILE);

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

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> fused_recurrent_kda(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale,
    const std::optional<ttnn::Tensor>& initial_state,
    const std::optional<ttnn::Tensor>& state_indices,
    const std::optional<ttnn::Tensor>& num_accepted_tokens,
    bool inplace_final_state,
    bool output_final_state,
    bool output_per_token_state,
    bool use_qk_l2norm_in_kernel,
    bool use_gate_in_kernel,
    const std::optional<ttnn::Tensor>& A_log,
    const std::optional<ttnn::Tensor>& dt_bias,
    std::optional<float> lower_bound,
    bool use_beta_sigmoid_in_kernel,
    bool allow_neg_eigval,
    const std::optional<ttnn::Tensor>& cu_seqlens,
    bool state_v_first,
    const std::optional<ttnn::Tensor>& output_tensor,
    const std::optional<ttnn::Tensor>& initial_state_block_idx,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<FusedRecurrentGdnProgramConfig>& program_config) {
    TT_FATAL(
        g.logical_shape().rank() == 4,
        "fused_recurrent_kda: g must be the per-key decay [B,T,HV,K] (rank 4), got {}",
        g.logical_shape());
    return fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        state_indices,
        num_accepted_tokens,
        inplace_final_state,
        output_final_state,
        output_per_token_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        lower_bound,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        cu_seqlens,
        state_v_first,
        output_tensor,
        initial_state_block_idx,
        memory_config,
        compute_kernel_config,
        program_config);
}

}  // namespace ttnn::transformer
