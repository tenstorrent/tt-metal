// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_ttnn.hpp"

#include <cmath>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "device/mhc_post_ttnn_device_operation.hpp"
#include "device/mhc_post_ttnn_device_operation_types.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

namespace {

constexpr uint32_t TILE_HW = 32;
constexpr int64_t MAX_STREAMS = 5;  // isqrt(TILE_HW): the n*n comb row must sit in one raw tile row (n*n <= 32)

std::vector<int64_t> dims(const Tensor& t) {
    const auto& s = t.logical_shape();
    std::vector<int64_t> d;
    for (size_t i = 0; i < s.rank(); ++i) {
        d.push_back(s[i]);
    }
    return d;
}

std::string dims_str(const std::vector<int64_t>& d) { return fmt::format("[{}]", fmt::join(d, ", ")); }

std::string dtype_name(DataType d) { return fmt::format("{}", d); }

// SUPPORTED (mhc_post.py): dtype / sublayer_dtype in {float32, bfloat16}, layout TILE, fp32_dest_acc_en True,
// alignment in {tile_aligned, h_non_aligned} (every T: the tagger has no refusal). EXCLUSIONS is empty.
void check_supported(const Tensor& f, const Tensor& x, const tt::tt_metal::ComputeConfigDescriptor& cfg) {
    auto float_types = [](DataType d) { return d == DataType::FLOAT32 || d == DataType::BFLOAT16; };
    if (!float_types(x.dtype())) {
        throw UnsupportedAxisError(fmt::format(
            "mhc_post: dtype={} not in SUPPORTED [DataType.FLOAT32, DataType.BFLOAT16]", dtype_name(x.dtype())));
    }
    if (!float_types(f.dtype())) {
        throw UnsupportedAxisError(fmt::format(
            "mhc_post: sublayer_dtype={} not in SUPPORTED [DataType.FLOAT32, DataType.BFLOAT16]",
            dtype_name(f.dtype())));
    }
    if (x.layout() != Layout::TILE) {
        throw UnsupportedAxisError("mhc_post: layout=ROW_MAJOR not in SUPPORTED [Layout.TILE]");
    }
    if (!cfg.fp32_dest_acc_en) {
        throw UnsupportedAxisError("mhc_post: fp32_dest_acc_en=False not in SUPPORTED [True]");
    }
}

void check_shapes(const Tensor& f, const Tensor& x, const Tensor& post, const Tensor& comb) {
    const std::vector<std::pair<const char*, const Tensor*>> named = {
        {"input_tensor", &f}, {"residual", &x}, {"post", &post}, {"comb", &comb}};
    for (const auto& [name, t] : named) {
        const auto d = dims(*t);
        if (d.size() < 2 || d.size() > 4) {
            throw ValueErrorCpp(fmt::format("mhc_post: {} rank must be 2..4, got {}", name, dims_str(d)));
        }
    }
    auto lead_of = [](const std::vector<int64_t>& d) { return std::vector<int64_t>(d.begin(), d.end() - 1); };
    const auto lead = lead_of(dims(f));
    for (const auto& [name, t] : named) {
        const auto l = lead_of(dims(*t));
        if (l != lead) {
            throw ValueErrorCpp(
                fmt::format("mhc_post: leading dims of {} {} != input_tensor's {}", name, dims_str(l), dims_str(lead)));
        }
    }
    const int64_t C = dims(f).back();
    const int64_t n = dims(post).back();
    if (C % TILE_HW != 0) {
        throw ValueErrorCpp(fmt::format("mhc_post: C={} must be a multiple of {}", C, TILE_HW));
    }
    if (n < 1 || n > MAX_STREAMS) {
        throw ValueErrorCpp(fmt::format("mhc_post: n={} must be in [1, {}]", n, MAX_STREAMS));
    }
    if (dims(x).back() != n * C) {
        throw ValueErrorCpp(fmt::format("mhc_post: residual last dim {} != n*C = {}", dims(x).back(), n * C));
    }
    if (dims(comb).back() != n * n) {
        throw ValueErrorCpp(fmt::format("mhc_post: comb last dim {} != n*n = {}", dims(comb).back(), n * n));
    }
    for (const auto& [name, t] : {std::pair{"post", &post}, std::pair{"comb", &comb}}) {
        if (t->dtype() != DataType::FLOAT32 || t->layout() != Layout::TILE) {
            throw ValueErrorCpp(fmt::format("mhc_post: {} must be float32 TILE", name));
        }
    }
    for (const auto& [name, t] : named) {
        if (t->layout() != Layout::TILE) {
            throw ValueErrorCpp(fmt::format("mhc_post: {} must be TILE_LAYOUT", name));
        }
        const auto& mc = t->memory_config();
        if (mc.memory_layout() != TensorMemoryLayout::INTERLEAVED || mc.buffer_type() != BufferType::DRAM) {
            throw ValueErrorCpp(fmt::format("mhc_post: {} must be DRAM interleaved", name));
        }
    }
}

}  // namespace

tt::tt_metal::ComputeConfigDescriptor default_compute_kernel_config() {
    tt::tt_metal::ComputeConfigDescriptor c;
    c.math_fidelity = MathFidelity::HiFi4;
    c.fp32_dest_acc_en = true;
    c.math_approx_mode = false;
    return c;
}

Tensor mhc_post(
    const Tensor& input_tensor,
    const Tensor& residual,
    const Tensor& post,
    const Tensor& comb,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    bool comb_transposed) {
    const auto cfg = compute_kernel_config.value_or(default_compute_kernel_config());
    check_supported(input_tensor, residual, cfg);
    check_shapes(input_tensor, residual, post, comb);
    return ttnn::prim::bringup::mhc_post_ttnn(input_tensor, residual, post, comb, cfg, comb_transposed);
}

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
