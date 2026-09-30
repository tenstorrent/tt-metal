// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_xing.hpp"

#include <algorithm>

#include <fmt/format.h>

#include "mhc_pre_ttnn.hpp"
#include "device/mhc_pre_ttnn_device_operation_types.hpp"
#include "device/mhc_pre_xing_device_operation.hpp"

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
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config) {
    const auto cfg = compute_kernel_config.value_or(default_compute_kernel_config());
    if (!cfg.fp32_dest_acc_en) {
        throw UnsupportedAxisError("mhc_pre_xing: fp32_dest_acc_en=False not in SUPPORTED [True]");
    }
    if (n < 1 || n > 4) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing: n={} must be in [1, 4]", n));
    }
    const int64_t ng = n * (n + 2);
    check_tile_dram_f32(input, "input");
    const int64_t in_last = input.logical_shape()[-1];
    if (coefficients_given) {
        if (in_last != ng) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: hc last dim {} != n (n + 2) = {}", in_last, ng));
        }
        if (!streams.has_value()) {
            throw ValueErrorCpp("mhc_pre_xing: coefficients_given needs streams (nothing to compute)");
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
        check_tile_dram_f32(*streams, "streams");
        const auto& ss = streams->logical_shape();
        const auto& is = input.logical_shape();
        if (ss.rank() != is.rank()) {
            throw ValueErrorCpp("mhc_pre_xing: streams and input ranks differ");
        }
        for (size_t i = 0; i + 1 < ss.rank(); ++i) {
            if (ss[i] != is[i]) {
                throw ValueErrorCpp("mhc_pre_xing: streams and input leading dims differ");
            }
        }
        if (ss[-1] % (n * TILE_HW) != 0) {
            throw ValueErrorCpp(fmt::format("mhc_pre_xing: streams last dim {} must be n*C with C % 32 == 0", ss[-1]));
        }
    }

    MhcPreXingParams p;
    p.n = static_cast<uint32_t>(n);
    p.compute_coef = !coefficients_given;
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
    p.compute_config = cfg;
    auto out = ttnn::prim::bringup::mhc_pre_xing(input, streams, p);
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

Tensor mhc_pre_xing_pack(const Tensor& mix, const Tensor& streams, int64_t n) {
    if (n < 1 || n > 4) {
        throw ValueErrorCpp(fmt::format("mhc_pre_xing_pack: n={} must be in [1, 4]", n));
    }
    check_tile_dram_f32(mix, "mix");
    check_tile_dram_f32(streams, "streams");
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
    return ttnn::prim::bringup::mhc_pre_xing(mix, streams, p).at(0);
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
