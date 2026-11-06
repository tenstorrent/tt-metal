// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Host side of fused_recurrent_gated_delta_rule: the contract (shape rules, then combination rules, in a fixed order
// so a given bad call always fails on the same message) in front of the program. The program reads the layer's
// tensors itself, folds the norm and the scale into its passes and writes the outputs in the caller's layouts; this
// file is the contract and one launch.

#include "fused_recurrent_gated_delta_rule.hpp"

#include <cmath>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include "device/fused_recurrent_gated_delta_rule_device_operation.hpp"

#include "ttnn/device.hpp"

using namespace tt::tt_metal;

namespace ttnn::transformer {

namespace {

constexpr const char* kOp = "fused_recurrent_gated_delta_rule";

// The dimensions a call names, read off q and v.
struct Dims {
    uint32_t B, T, H, K, HV, V;
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

// Shape rules: ranks, consistent dimensions, dtypes and layouts, head dimensions, the token cap.
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
    TT_FATAL(
        gs.rank() == 3 && gs[0] == d.B && gs[1] == d.T && gs[2] == d.HV, "{}: g must be [B,T,HV], got {}", kOp, gs);
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
        TT_FATAL(t->layout() == Layout::TILE, "{}: {} must be TILE layout", kOp, name);
        TT_FATAL(!t->is_sharded(), "{}: {} must be interleaved, not sharded", kOp, name);
    }
    TT_FATAL(
        k.dtype() == q.dtype() && v.dtype() == q.dtype(),
        "{}: q, k and v must share one dtype (bfloat16 or float32), got q {} k {} v {}",
        kOp,
        q.dtype(),
        k.dtype(),
        v.dtype());
    return d;
}

// Combination rules. Returns Bs, the rows of the state tensor (B when there is none).
uint32_t check_rules(
    const Dims& d,
    const std::optional<ttnn::Tensor>& initial_state,
    bool inplace_final_state,
    bool output_per_token_state,
    const std::optional<ttnn::Tensor>& output_tensor,
    const std::optional<ttnn::Tensor>& initial_state_block_idx,
    const std::optional<FusedRecurrentGdnProgramConfig>& program_config,
    const ttnn::Tensor& q) {
    const bool ring = initial_state_block_idx.has_value();
    uint32_t Bs = d.B;
    // The direct state: [Bs, HV, K, V] fp32 with Bs >= B (the leading-row form). The ring's shape is the device
    // operation's check.
    if (initial_state.has_value()) {
        require_on_device(*initial_state, "initial_state", q);
        if (!ring) {
            const auto& ss = initial_state->logical_shape();
            TT_FATAL(
                ss.rank() == 4 && static_cast<uint32_t>(ss[0]) >= d.B && ss[1] == d.HV && ss[2] == d.K &&
                    ss[3] == d.V && initial_state->dtype() == DataType::FLOAT32,
                "{}: initial_state must be [Bs, HV, K, V] fp32 with Bs >= B (rows [0:B] are the batch), got {} {}",
                kOp,
                initial_state->dtype(),
                ss);
            Bs = ss[0];
        }
    }
    TT_FATAL(!inplace_final_state || initial_state.has_value(), "{}: inplace_final_state requires initial_state", kOp);
    TT_FATAL(
        !(inplace_final_state && ring),
        "{}: inplace_final_state does not apply in ring mode (the ring is written in place already)",
        kOp);
    TT_FATAL(
        !(inplace_final_state && output_per_token_state),
        "{}: inplace_final_state and output_per_token_state are exclusive",
        kOp);
    // The preallocated output: o's shape in the input dtype.
    if (output_tensor.has_value()) {
        const auto& os = output_tensor->logical_shape();
        const bool ok = os.rank() == 4 && os[0] == d.B && os[1] == d.T && os[2] == d.HV && os[3] == d.V &&
                        output_tensor->dtype() == q.dtype() && output_tensor->layout() == Layout::TILE &&
                        output_tensor->storage_type() == StorageType::DEVICE && output_tensor->device() == q.device();
        TT_FATAL(
            ok,
            "{}: output_tensor must be [B, T, HV, V] in the input dtype, TILE, on q's device, got {} {}",
            kOp,
            output_tensor->dtype(),
            os);
    }
    TT_FATAL(!ring || output_per_token_state, "{}: initial_state_block_idx requires output_per_token_state", kOp);
    // The geometry: the fields are validated as the contract defines them, and this build serves the defaults.
    const uint32_t num_slices = program_config.has_value() ? program_config->num_slices.value_or(1u) : 1u;
    const uint32_t units_per_core = program_config.has_value() ? program_config->units_per_core.value_or(1u) : 1u;
    TT_FATAL(
        num_slices == 1 || num_slices == 2 || num_slices == 4,
        "{}: program_config.num_slices must be 1, 2 or 4, got {}",
        kOp,
        num_slices);
    TT_FATAL(units_per_core >= 1, "{}: program_config.units_per_core must be at least 1, got {}", kOp, units_per_core);
    TT_FATAL(
        num_slices == 1 && units_per_core == 1,
        "{}: this build runs one whole head per core: program_config.num_slices must be 1 and units_per_core 1, got "
        "{} and {}",
        kOp,
        num_slices,
        units_per_core);
    return Bs;
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
    bool inplace_final_state,
    bool output_final_state,
    bool output_per_token_state,
    bool use_qk_l2norm_in_kernel,
    const std::optional<ttnn::Tensor>& output_tensor,
    const std::optional<ttnn::Tensor>& initial_state_block_idx,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<FusedRecurrentGdnProgramConfig>& program_config) {
    const Dims d = check_shapes(q_in, k_in, v_in, g_in, beta_in);
    const uint32_t Bs = check_rules(
        d,
        initial_state,
        inplace_final_state,
        output_per_token_state,
        output_tensor,
        initial_state_block_idx,
        program_config,
        q_in);

    auto* dev = q_in.device();
    const uint32_t B = d.B;
    const uint32_t T = d.T;
    const uint32_t H = d.H;
    const uint32_t K = d.K;
    const uint32_t HV = d.HV;
    const float scale = scale_opt.has_value() ? *scale_opt : (1.0f / std::sqrt(static_cast<float>(K)));
    (void)Bs;
    const auto kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    // Everything reaches the program as the layer holds it: the reader gathers the rows and does the GQA mapping, the
    // compute kernel folds the optional norm and the scale into its passes, the writer emits o in the input dtype
    // into [B,T,HV,V] (or the caller's output_tensor) and the state into [B,HV,K,V], [B,T,HV,K,V], the ring, or the
    // caller's state itself (inplace_final_state). The state is read as the caller holds it: [Bs,HV,K,V] fp32 with
    // Bs >= B (rows [0:B] are the batch; head h's tiles are its pages h*kv..), the [T*BH,K,V] ring in ring mode, or
    // absent, in which case the reader builds a zero state.
    const bool ring = initial_state_block_idx.has_value();
    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const bool want_state = output_final_state || output_per_token_state || inplace_final_state;
    auto res = ttnn::prim::fused_recurrent_gated_delta_rule(
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        initial_state,
        initial_state_block_idx,
        output_tensor,
        B,
        H,
        HV,
        T,
        want_state && !output_per_token_state,
        output_per_token_state,
        inplace_final_state,
        use_qk_l2norm_in_kernel,
        scale,
        out_mem,
        kernel_cfg);
    // res[0] is o (the caller's output_tensor when given); res[1] the state in the requested layout: the ring itself
    // (ring mode), the caller's state (in place), [B,T,HV,K,V] (per token) or [B,HV,K,V] (final).
    std::optional<ttnn::Tensor> state_opt;
    if (ring || inplace_final_state || output_per_token_state || output_final_state) {
        state_opt = res[1];
    }
    return {res[0], state_opt};
}

}  // namespace ttnn::transformer
