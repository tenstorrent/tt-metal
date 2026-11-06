// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Host side of fused_recurrent_gated_delta_rule: the contract (shape rules, then combination rules, in a fixed order
// so a given bad call always fails on the same message) in front of the program. The program still takes the
// per-token tile layout of step 1a, so this file also performs the relayout chain around it; the following commits
// of step 1b move each piece of that chain into the kernels and delete it here.

#include "fused_recurrent_gated_delta_rule.hpp"

#include <cmath>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include "device/fused_recurrent_gated_delta_rule_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
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

// use_qk_l2norm_in_kernel: x / sqrt(sum_K x^2 + eps) in fp32 with eps inside the root (the golden's l2_norm), as
// host ops in front of the program until the compute commit folds it into the constants pass.
ttnn::Tensor l2_norm_rows(const ttnn::Tensor& x_in, const DeviceComputeKernelConfig& cfg) {
    const ttnn::Tensor x = as_f32(x_in);
    const ttnn::Tensor sum_sq = ttnn::sum(ttnn::multiply(x, x), -1, /*keepdim=*/true, std::nullopt, cfg);
    const ttnn::Tensor inv = ttnn::rsqrt(ttnn::add(sum_sq, 1e-6f));
    return ttnn::multiply(x, inv);
}

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

    // g (log decay) as-is: the kernel computes exp(g) on the SFPU; beta as-is (already sigmoid'd).
    ttnn::Tensor g = headvec_split(g_in, B, T, HV);        // [BH, T]
    ttnn::Tensor beta = headvec_split(beta_in, B, T, HV);  // [BH, T]

    // Per-token layout: each (head, token) its own zero-padded tile-block.
    //   q/k [BH*T,1,K], v [BH*T,1,V], g/beta [BH*T,1,1].
    // T==1 fast path: head_split already yields [BH,1,D] == [BH*T,1,D], so the q/k/v relayout is a
    // no-op — skip the RM round-trip (it is ~5x the cost of the permute; see test_headsplit_bench).
    if (T > 1) {
        q = reshape_rm(q, ttnn::Shape({BH * T, 1, K}));
        k = reshape_rm(k, ttnn::Shape({BH * T, 1, K}));
        v = reshape_rm(v, ttnn::Shape({BH * T, 1, V}));
    }
    // g/beta come from headvec_split as [BH,T] (BH packed in tile-rows); they always need the
    // relayout to per-scalar tiles [BH*T,1,1] so the reader finds the value at element [0,0]. They are tiny.
    g = reshape_rm(g, ttnn::Shape({BH * T, 1, 1}));
    beta = reshape_rm(beta, ttnn::Shape({BH * T, 1, 1}));

    // Initial state: rows [0:B] of [Bs,HV,K,V] -> [BH,K,V] fp32 TILE. Always provide (zeros if absent) so the reader
    // always reads S (the unconditional path; there is no in-kernel zeroing yet). Traced callers pass a persistent
    // state buffer (never absent); the zeros() fallback here is eager-only (device-side fill, uncached).
    //
    // Ring mode: initial_state is the [T*BH,K,V] ring and also the state output; pass it through untouched, since a
    // reshape or typecast would break the in-place aliasing. The device op validates it.
    // In place: the device operation takes the caller's [Bs,HV,K,V] tensor as it is and aliases its state output
    // onto it (head h's tiles are its pages h*kv.., rows [0:B]).
    const bool ring = initial_state_block_idx.has_value();
    std::optional<ttnn::Tensor> s0 = initial_state;
    if (!ring && !inplace_final_state) {
        if (initial_state.has_value()) {
            ttnn::Tensor rows = *initial_state;
            if (Bs != B) {
                rows = ttnn::slice(
                    rows,
                    ttsl::SmallVector<uint32_t>{0, 0, 0, 0},
                    ttsl::SmallVector<uint32_t>{B, HV, K, V},
                    ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
            }
            s0 = ttnn::reshape(rows, ttnn::Shape({BH, K, V}));
        } else {
            s0 = ttnn::zeros(
                ttnn::Shape({BH, K, V}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
        }
    }

    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const bool want_state = output_final_state || output_per_token_state || inplace_final_state;
    auto res = ttnn::prim::fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        s0,
        initial_state_block_idx,
        T,
        want_state && !output_per_token_state,
        output_per_token_state,
        inplace_final_state,
        out_mem,
        kernel_cfg);
    const ttnn::Tensor& o_pt = res[0];  // [BH*T, 1, V]
    const ttnn::Tensor& st = res[1];    // [BH*T,K,V] per-token, else [BH,K,V]

    // o [BH*T,1,V] -> [B,HV,T,V] -> [B,T,HV,V], in the input dtype, into the caller's buffer when given.
    ttnn::Tensor o = ttnn::to_layout(o_pt, Layout::ROW_MAJOR);
    o = ttnn::reshape(o, ttnn::Shape({B, HV, T, V}));
    o = ttnn::permute(o, ttsl::SmallVector<int64_t>{0, 2, 1, 3});  // [B,T,HV,V]
    o = ttnn::to_layout(o, Layout::TILE);
    if (q_in.dtype() != DataType::FLOAT32) {
        o = ttnn::typecast(o, q_in.dtype());
    }
    if (output_tensor.has_value()) {
        ttnn::copy(o, *output_tensor);
        o = *output_tensor;
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
    } else if (inplace_final_state) {
        // res[1] is the caller's state tensor: head h's final state landed in its pages, i.e. rows [0:B].
        state_opt = st;
    } else if (output_final_state) {
        state_opt = ttnn::reshape(st, ttnn::Shape({B, HV, K, V}));
    }

    return {o, state_opt};
}

}  // namespace ttnn::transformer
