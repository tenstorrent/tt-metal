// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_xing.hpp"

#include <algorithm>

#include <fmt/format.h>

#include "mhc_pre_ttnn.hpp"
#include "device/mhc_pre_ttnn_device_operation_types.hpp"
#include "device/mhc_pre_xing_device_operation.hpp"
#include "device/mhc_engram_pack_device_operation.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

namespace {

constexpr int64_t TILE_HW = 32;

void check_tile_dram_f32(const Tensor& t, const char* name) {
    if (t.dtype() != DataType::FLOAT32 || t.layout() != Layout::TILE) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing: {} must be float32 TILE", name));
    }
    const auto& mc = t.memory_config();
    if (mc.memory_layout() != TensorMemoryLayout::INTERLEAVED || mc.buffer_type() != BufferType::DRAM) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing: {} must be DRAM interleaved", name));
    }
    if (t.logical_shape().rank() < 2 || t.logical_shape().rank() > 4) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing: {} rank must be 2..4", name));
    }
}

// The streams (and the engram keys): float32, or (deepseek_v41_flash_d_p P.20) bfloat16 TILE; a bf16 y / X' follows.
void check_tile_dram_stream(const Tensor& t, const char* name) {
    if (t.dtype() == DataType::BFLOAT16) {
        if (t.layout() != Layout::TILE) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: {} must be float32 or bfloat16 TILE", name));
        }
        const auto& mc = t.memory_config();
        if (mc.memory_layout() != TensorMemoryLayout::INTERLEAVED || mc.buffer_type() != BufferType::DRAM) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: {} must be DRAM interleaved", name));
        }
        if (t.logical_shape().rank() < 2 || t.logical_shape().rank() > 4) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: {} rank must be 2..4", name));
        }
        return;
    }
    check_tile_dram_f32(t, name);
}

}  // namespace

std::tuple<std::optional<Tensor>, std::optional<Tensor>> mhc_pre_xing(
    const Tensor& input,
    const std::optional<Tensor>& streams,
    const std::vector<double>& scale,
    const std::vector<double>& base,
    double norm_width,
    int64_t n,
    double norm_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    double clamp_min,
    double clamp_max,
    bool coefficients_given,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    bool deepseek,
    const std::optional<Tensor>& collapse_pre,
    int64_t partials,
    const std::optional<Tensor>& hc_output,
    const std::optional<Tensor>& y_output) {
    const auto cfg = compute_kernel_config.value_or(default_compute_kernel_config());
    if (!cfg.fp32_dest_acc_en) {
        throw UnsupportedAxisError("mhc_pre_xing: fp32_dest_acc_en=False not in SUPPORTED [True]");
    }
    if (n < 1 || n > 4) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing: n={} must be in [1, 4]", n));
    }
    const int64_t ng = n * (n + 2);
    check_tile_dram_f32(input, "input");
    if (partials < 1 || partials > 16) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing: partials={} must be in [1, 16]", partials));
    }
    // partials > 1: input (1, P, T, 32); the leading-dim checks below compare its dim -3 as 1
    const bool stacked = partials > 1;
    if (stacked) {
        if (coefficients_given) {
            throw ValueErrorCpp("mhc_pre_xing: partials needs the mix row (coefficients_given=False)");
        }
        const auto& is = input.logical_shape();
        if (is.rank() != 4 || is[0] != 1 || is[1] != partials || is[2] % TILE_HW != 0) {
            throw ValueErrorCpp(fmt::format(
                "mhc_pre_xing: partials={} needs input (1, {}, T, 32) with T % 32 == 0", partials, partials));
        }
    }
    auto lead_of_input = [&](size_t i) -> int64_t {
        return (stacked && i + 3 == input.logical_shape().rank()) ? 1 : input.logical_shape()[i];
    };
    const int64_t in_last = input.logical_shape()[-1];
    if (coefficients_given) {
        if (in_last != ng) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: hc last dim {} != n (n + 2) = {}", in_last, ng));
        }
        if (!streams.has_value()) {
            throw ValueErrorCpp("mhc_pre_xing: coefficients_given needs streams (nothing to compute)");
        }
        if (deepseek) {
            throw ValueErrorCpp("mhc_pre_xing: deepseek selects the coefficient math; coefficients_given has none");
        }
    } else {
        if (in_last != TILE_HW) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: mix row last dim {} != 32", in_last));
        }
        if (scale.size() != 3) {
            throw ValueErrorCpp("mhc_pre_xing: scale must be (a_pre, a_post, a_res)");
        }
        if (static_cast<int64_t>(base.size()) != ng) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: base has {} values, want {}", base.size(), ng));
        }
        if (!(norm_width > 0.0)) {
            throw ValueErrorCpp("mhc_pre_xing: norm_width must be > 0");
        }
        if (sinkhorn_iters < 1) {
            throw ValueErrorCpp("mhc_pre_xing: sinkhorn_iters must be >= 1");
        }
    }
    if (streams.has_value()) {
        check_tile_dram_stream(*streams, "streams");
        const auto& ss = streams->logical_shape();
        const auto& is = input.logical_shape();
        if (ss.rank() != is.rank()) {
            throw ValueErrorCpp("mhc_pre_xing: streams and input ranks differ");
        }
        for (size_t i = 0; i + 1 < ss.rank(); ++i) {
            if (ss[i] != (i + 1 < is.rank() ? lead_of_input(i) : is[i])) {
                throw ValueErrorCpp("mhc_pre_xing: streams and input leading dims differ");
            }
        }
        if (ss[-1] % (n * TILE_HW) != 0) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: streams last dim {} must be n*C with C % 32 == 0", ss[-1]));
        }
    }

    if (collapse_pre.has_value()) {
        if (coefficients_given || !streams.has_value()) {
            throw ValueErrorCpp(
                "mhc_pre_xing: collapse_pre needs streams and the coefficients (coefficients_given=False)");
        }
        check_tile_dram_f32(*collapse_pre, "collapse_pre");
        const auto& ps = collapse_pre->logical_shape();
        const auto& is = input.logical_shape();
        if (ps.rank() != is.rank()) {
            throw ValueErrorCpp("mhc_pre_xing: collapse_pre and input ranks differ");
        }
        for (size_t i = 0; i + 1 < ps.rank(); ++i) {
            if (ps[i] != (i + 1 < is.rank() ? lead_of_input(i) : is[i])) {
                throw ValueErrorCpp("mhc_pre_xing: collapse_pre and input leading dims differ");
            }
        }
        if (ps[-1] < n || ps[-1] > TILE_HW) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: collapse_pre last dim {} must be in [n, 32]", ps[-1]));
        }
    }

    MhcPreXingParams p;
    p.n = static_cast<uint32_t>(n);
    p.compute_coef = !coefficients_given;
    p.given_pre = collapse_pre.has_value();
    p.partials = static_cast<uint32_t>(partials);
    if (!coefficients_given) {
        p.scale = {scale[0], scale[1], scale[2]};
        p.base = base;
        p.inv_nc = 1.0 / norm_width;
    }
    p.norm_eps = norm_eps;
    p.hc_eps = hc_eps;
    p.sinkhorn_iters = static_cast<uint32_t>(std::max<int64_t>(sinkhorn_iters, 1));
    p.clamp_min = clamp_min;
    p.clamp_max = clamp_max;
    p.deepseek = deepseek;
    p.compute_config = cfg;
    auto out = ttnn::prim::bringup::mhc_pre_xing(input, streams, p, collapse_pre, hc_output, y_output);
    std::optional<Tensor> hc, y;
    size_t k = 0;
    if (p.compute_coef) {
        hc = out.at(k++);
    }
    if (streams.has_value()) {
        y = out.at(k++);
    }
    return {hc, y};
}

Tensor mhc_pre_xing_pack(
    const std::optional<Tensor>& mix_opt,
    const Tensor& streams,
    int64_t n,
    const std::optional<Tensor>& engram_keys,
    const std::optional<Tensor>& engram_weight,
    const std::optional<Tensor>& output_tensor) {
    if (n < 1 || n > 4) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing_pack: n={} must be in [1, 4]", n));
    }
    if (engram_keys.has_value() || engram_weight.has_value()) {
        if (!engram_keys.has_value() || !engram_weight.has_value() || mix_opt.has_value()) {
            throw ValueErrorCpp("mhc_pre_xing_pack: engram mode needs engram_keys and engram_weight, and mix None");
        }
        if (output_tensor.has_value()) {
            throw ValueErrorCpp("mhc_pre_xing_pack: output_tensor is not supported in engram mode");
        }
        check_tile_dram_stream(streams, "streams");
        check_tile_dram_stream(*engram_keys, "engram_keys");
        check_tile_dram_f32(*engram_weight, "engram_weight");
        if (engram_keys->dtype() != streams.dtype()) {
            throw ValueErrorCpp("mhc_pre_xing_pack: engram_keys and streams must share a dtype (float32 or bfloat16)");
        }
        const auto& ss = streams.logical_shape();
        const auto& ks = engram_keys->logical_shape();
        const auto& ws = engram_weight->logical_shape();
        if (ss[-1] % (n * TILE_HW) != 0) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing_pack: streams last dim {} must be n*C, C % 32 == 0", ss[-1]));
        }
        if (ks.rank() != ss.rank() || ks[-1] < ss[-1]) {
            throw ValueErrorCpp("mhc_pre_xing_pack: engram_keys must be (..., T, >= n*C) like the streams");
        }
        for (size_t i = 0; i + 1 < ss.rank(); ++i) {
            if (ks[i] != ss[i]) {
                throw ValueErrorCpp("mhc_pre_xing_pack: engram_keys and streams leading dims differ");
            }
        }
        int64_t lead = 1;
        for (size_t i = 0; i + 2 < ws.rank(); ++i) {
            lead *= ws[i];
        }
        if (lead != 1 || ws[-2] != TILE_HW || ws[-1] != ss[-1]) {
            throw ValueErrorCpp("mhc_pre_xing_pack: engram_weight must be (..., 32, n*C) with leading dims 1");
        }
        return ttnn::prim::bringup::mhc_engram_pack(streams, *engram_keys, *engram_weight, static_cast<uint32_t>(n))
            .at(0);  // [1] is the partials scratch, freed with the vector
    }
    if (!mix_opt.has_value()) {
        throw ValueErrorCpp("mhc_pre_xing_pack: mix is required (it is None only in engram mode)");
    }
    const Tensor& mix = *mix_opt;
    check_tile_dram_f32(mix, "mix");
    check_tile_dram_stream(streams, "streams");
    if (mix.logical_shape()[-1] != TILE_HW) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing_pack: mix last dim {} != 32", mix.logical_shape()[-1]));
    }
    const auto& ss = streams.logical_shape();
    const auto& ms = mix.logical_shape();
    if (ss.rank() != ms.rank()) {
        throw ValueErrorCpp("mhc_pre_xing_pack: streams and mix ranks differ");
    }
    for (size_t i = 0; i + 1 < ss.rank(); ++i) {
        if (ss[i] != ms[i]) {
            throw ValueErrorCpp("mhc_pre_xing_pack: streams and mix leading dims differ");
        }
    }
    if (ss[-1] % (n * TILE_HW) != 0) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing_pack: streams last dim {} must be n*C, C % 32 == 0", ss[-1]));
    }
    MhcPreXingParams p;
    p.n = static_cast<uint32_t>(n);
    p.compute_coef = false;
    p.pack_stats = true;
    p.compute_config = default_compute_kernel_config();
    return ttnn::prim::bringup::mhc_pre_xing(mix, streams, p, std::nullopt, output_tensor).at(0);
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
