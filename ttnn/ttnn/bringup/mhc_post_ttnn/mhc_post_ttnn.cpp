// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_ttnn.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "device/mhc_post_ttnn_device_operation.hpp"
#include "device/mhc_post_stats_device_operation.hpp"
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

// post_col / comb_col (fork option, deepseek_v41_flash_d_p P.19): post_k is post's column post_col + k, comb[i*n + j]
// comb's column comb_col + i*n + j (both may be the packed [pre | post | comb] mixes); n is then the residual's
// stream count (residual last dim / C). Both 0 (default): n = post's last dim, the checks as they always were.
void check_shapes(
    const Tensor& f,
    const Tensor& x,
    const Tensor& post,
    const Tensor& comb,
    int64_t post_col = 0,
    int64_t comb_col = 0) {
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
    const bool offsets = post_col != 0 || comb_col != 0;
    if (C % TILE_HW != 0) {
        throw ValueErrorCpp(fmt::format("mhc_post: C={} must be a multiple of {}", C, TILE_HW));
    }
    const int64_t n = offsets ? dims(x).back() / C : dims(post).back();
    if (offsets) {
        if (post_col < 0 || comb_col < 0 || post_col + n > TILE_HW || comb_col + n * n > TILE_HW) {
            throw ValueErrorCpp(fmt::format(
                "mhc_post: post_col={} / comb_col={} with n={} must keep post and comb in one tile row",
                post_col,
                comb_col,
                n));
        }
        if (dims(post).back() < post_col + n || dims(comb).back() < comb_col + n * n) {
            throw ValueErrorCpp(fmt::format(
                "mhc_post: post last dim {} < post_col + n = {} or comb last dim {} < comb_col + n*n = {}",
                dims(post).back(),
                post_col + n,
                dims(comb).back(),
                comb_col + n * n));
        }
    }
    if (n < 1 || n > MAX_STREAMS) {
        throw ValueErrorCpp(fmt::format("mhc_post: n={} must be in [1, {}]", n, MAX_STREAMS));
    }
    if (dims(x).back() != n * C) {
        throw ValueErrorCpp(fmt::format("mhc_post: residual last dim {} != n*C = {}", dims(x).back(), n * C));
    }
    if (!offsets && dims(comb).back() != n * n) {
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
    c.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
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
    bool comb_transposed,
    int64_t post_col,
    int64_t comb_col,
    const std::optional<Tensor>& output_tensor,
    std::optional<tt::tt_metal::CoreCoord> max_grid) {
    const auto cfg = compute_kernel_config.value_or(default_compute_kernel_config());
    check_supported(input_tensor, residual, cfg);
    check_shapes(input_tensor, residual, post, comb, post_col, comb_col);
    return ttnn::prim::bringup::mhc_post_ttnn(
        input_tensor,
        residual,
        post,
        comb,
        cfg,
        comb_transposed,
        static_cast<uint32_t>(post_col),
        static_cast<uint32_t>(comb_col),
        output_tensor,
        max_grid);
}

std::tuple<Tensor, Tensor> mhc_post_with_stats(
    const Tensor& input_tensor,
    const Tensor& residual,
    const Tensor& post,
    const Tensor& comb,
    const Tensor& stats_weight,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    bool comb_transposed,
    int64_t post_col,
    int64_t comb_col,
    const std::optional<Tensor>& output_tensor,
    const std::optional<Tensor>& stats_output,
    const std::optional<Tensor>& stats_scratch,
    std::optional<tt::tt_metal::CoreCoord> max_grid) {
    const auto cfg = compute_kernel_config.value_or(default_compute_kernel_config());
    check_supported(input_tensor, residual, cfg);
    check_shapes(input_tensor, residual, post, comb, post_col, comb_col);
    // F float32, or bfloat16 (fork option, deepseek_v41_flash_d_p P.19: unpacked to the fp32 DEST exactly); the
    // residual float32, or bfloat16 with a bfloat16 F (P.20: X' bf16, the next hc row from the bf16 X')
    const bool f_ok = input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16;
    const bool x_ok = residual.dtype() == DataType::FLOAT32 ||
                      (residual.dtype() == DataType::BFLOAT16 && input_tensor.dtype() == DataType::BFLOAT16);
    if (!f_ok || !x_ok) {
        throw ValueErrorCpp(
            "mhc_post: stats_weight needs a float32 residual with a float32 or bfloat16 input_tensor, or a bfloat16 "
            "residual with a bfloat16 input_tensor");
    }
    const int64_t C = dims(input_tensor).back();
    const int64_t n = dims(residual).back() / C;
    if (n > 4) {
        throw ValueErrorCpp(fmt::format("mhc_post: stats_weight needs n <= 4 (column n (n + 2) < 32), got {}", n));
    }
    const auto w = dims(stats_weight);
    int64_t lead = 1;
    for (size_t i = 0; i + 2 < w.size(); ++i) {
        lead *= w[i];
    }
    if (w.size() < 2 || w.size() > 4 || lead != 1 || w[w.size() - 2] != n * C || w.back() != TILE_HW) {
        throw ValueErrorCpp(fmt::format(
            "mhc_post: stats_weight must be (..., n*C = {}, 32) with leading dims 1, got {}", n * C, dims_str(w)));
    }
    const auto& mc = stats_weight.memory_config();
    if (stats_weight.dtype() != DataType::FLOAT32 || stats_weight.layout() != Layout::TILE ||
        mc.memory_layout() != TensorMemoryLayout::INTERLEAVED || mc.buffer_type() != BufferType::DRAM) {
        throw ValueErrorCpp("mhc_post: stats_weight must be float32 TILE, DRAM interleaved");
    }
    auto out = ttnn::prim::bringup::mhc_post_stats(
        input_tensor,
        residual,
        post,
        comb,
        stats_weight,
        cfg,
        comb_transposed,
        static_cast<uint32_t>(post_col),
        static_cast<uint32_t>(comb_col),
        output_tensor,
        stats_output,
        stats_scratch,
        max_grid);
    return {out.at(0), out.at(1)};  // out[2] is the partials scratch, freed with the vector
}

std::tuple<Tensor, Tensor, Tensor, Tensor> mhc_post_with_collapse(
    const Tensor& input_tensor,
    const Tensor& residual,
    const Tensor& post,
    const Tensor& comb,
    const Tensor& stats_weight,
    const Tensor& collapse_pre,
    const std::vector<double>& hc_scale,
    const std::vector<double>& hc_base,
    double hc_norm_width,
    double hc_norm_eps,
    double hc_eps,
    int64_t hc_sinkhorn_iters,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    bool comb_transposed,
    int64_t post_col,
    int64_t comb_col,
    const std::optional<Tensor>& output_tensor,
    const std::optional<Tensor>& stats_output,
    const std::optional<Tensor>& stats_scratch,
    const std::optional<Tensor>& hc_output,
    const std::optional<Tensor>& y_output,
    std::optional<tt::tt_metal::CoreCoord> max_grid,
    const std::optional<Tensor>& norm_weight,
    double norm_eps) {
    // the stats_weight checks first (the same refusals), then the collapse's own
    const auto cfg = compute_kernel_config.value_or(default_compute_kernel_config());
    check_supported(input_tensor, residual, cfg);
    check_shapes(input_tensor, residual, post, comb, post_col, comb_col);
    if (residual.dtype() != DataType::BFLOAT16 || input_tensor.dtype() != DataType::BFLOAT16) {
        throw ValueErrorCpp("mhc_post: collapse_pre needs a bfloat16 residual and a bfloat16 input_tensor");
    }
    const int64_t C = dims(input_tensor).back();
    const int64_t n = dims(residual).back() / C;
    if (n > 4) {
        throw ValueErrorCpp(fmt::format("mhc_post: collapse_pre needs n <= 4, got {}", n));
    }
    const auto pd = dims(collapse_pre);
    const auto fd = dims(input_tensor);
    if (pd.size() != fd.size() || !std::equal(pd.begin(), pd.end() - 1, fd.begin()) || pd.back() < n ||
        pd.back() > TILE_HW) {
        throw ValueErrorCpp(fmt::format(
            "mhc_post: collapse_pre must be (..., T, n..32) with input_tensor's leading dims, got {}", dims_str(pd)));
    }
    const auto& pmc = collapse_pre.memory_config();
    if (collapse_pre.dtype() != DataType::FLOAT32 || collapse_pre.layout() != Layout::TILE ||
        pmc.memory_layout() != TensorMemoryLayout::INTERLEAVED || pmc.buffer_type() != BufferType::DRAM) {
        throw ValueErrorCpp("mhc_post: collapse_pre must be float32 TILE, DRAM interleaved");
    }
    if (hc_scale.size() != 3) {
        throw ValueErrorCpp("mhc_post: hc_scale must be (a_pre, a_post, a_res)");
    }
    if (static_cast<int64_t>(hc_base.size()) != n * (n + 2)) {
        throw ValueErrorCpp(fmt::format("mhc_post: hc_base has {} values, want {}", hc_base.size(), n * (n + 2)));
    }
    if (!(hc_norm_width > 0.0)) {
        throw ValueErrorCpp("mhc_post: hc_norm_width must be > 0");
    }
    if (hc_sinkhorn_iters < 1) {
        throw ValueErrorCpp("mhc_post: hc_sinkhorn_iters must be >= 1");
    }
    const auto w = dims(stats_weight);
    int64_t lead = 1;
    for (size_t i = 0; i + 2 < w.size(); ++i) {
        lead *= w[i];
    }
    if (w.size() < 2 || w.size() > 4 || lead != 1 || w[w.size() - 2] != n * C || w.back() != TILE_HW) {
        throw ValueErrorCpp(fmt::format(
            "mhc_post: stats_weight must be (..., n*C = {}, 32) with leading dims 1, got {}", n * C, dims_str(w)));
    }
    const auto& mc = stats_weight.memory_config();
    if (stats_weight.dtype() != DataType::FLOAT32 || stats_weight.layout() != Layout::TILE ||
        mc.memory_layout() != TensorMemoryLayout::INTERLEAVED || mc.buffer_type() != BufferType::DRAM) {
        throw ValueErrorCpp("mhc_post: stats_weight must be float32 TILE, DRAM interleaved");
    }
    if (norm_weight.has_value()) {  // P.35: (..., 1, C) fp32 TILE DRAM, leading dims 1
        const auto nd = dims(*norm_weight);
        int64_t vol = 1;
        for (const auto d : nd) {
            vol *= d;
        }
        const auto& nmc = norm_weight->memory_config();
        if (nd.empty() || nd.back() != C || vol != C || norm_weight->dtype() != DataType::FLOAT32 ||
            norm_weight->layout() != Layout::TILE || nmc.memory_layout() != TensorMemoryLayout::INTERLEAVED ||
            nmc.buffer_type() != BufferType::DRAM) {
            throw ValueErrorCpp(fmt::format(
                "mhc_post: norm_weight must be (..., 1, C = {}) float32 TILE, DRAM interleaved, got {}",
                C,
                dims_str(nd)));
        }
        if (!(norm_eps >= 0.0)) {
            throw ValueErrorCpp("mhc_post: norm_eps must be >= 0");
        }
    }
    MhcPostCollapseArgs ca{
        .collapse_pre = collapse_pre,
        .scale = {hc_scale[0], hc_scale[1], hc_scale[2]},
        .base = hc_base,
        .inv_nc = 1.0 / hc_norm_width,
        .norm_eps = hc_norm_eps,
        .hc_eps = hc_eps,
        .sinkhorn_iters = static_cast<uint32_t>(hc_sinkhorn_iters),
        .hc_output = hc_output,
        .y_output = y_output,
        .norm_weight = norm_weight,
        .rms_eps = norm_eps};
    auto out = ttnn::prim::bringup::mhc_post_stats(
        input_tensor,
        residual,
        post,
        comb,
        stats_weight,
        cfg,
        comb_transposed,
        static_cast<uint32_t>(post_col),
        static_cast<uint32_t>(comb_col),
        output_tensor,
        stats_output,
        stats_scratch,
        max_grid,
        ca);
    return {out.at(0), out.at(1), out.at(3), out.at(4)};  // out[2] is the partials scratch
}

std::variant<Tensor, std::tuple<Tensor, Tensor>, std::tuple<Tensor, Tensor, Tensor, Tensor>> mhc_post_py(
    const Tensor& input_tensor,
    const Tensor& residual,
    const Tensor& post,
    const Tensor& comb,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    bool comb_transposed,
    const std::optional<Tensor>& stats_weight,
    int64_t post_col,
    int64_t comb_col,
    const std::optional<Tensor>& output_tensor,
    const std::optional<Tensor>& stats_output,
    const std::optional<Tensor>& stats_scratch,
    std::optional<tt::tt_metal::CoreCoord> max_grid,
    const std::optional<Tensor>& collapse_pre,
    const std::optional<std::vector<double>>& hc_scale,
    const std::optional<std::vector<double>>& hc_base,
    double hc_norm_width,
    double hc_norm_eps,
    double hc_eps,
    int64_t hc_sinkhorn_iters,
    const std::optional<Tensor>& hc_output,
    const std::optional<Tensor>& y_output,
    const std::optional<Tensor>& norm_weight,
    double norm_eps) {
    if (collapse_pre.has_value()) {  // P.34
        if (!stats_weight.has_value() || !hc_scale.has_value() || !hc_base.has_value()) {
            throw ValueErrorCpp("mhc_post: collapse_pre needs stats_weight, hc_scale and hc_base");
        }
        return mhc_post_with_collapse(
            input_tensor,
            residual,
            post,
            comb,
            *stats_weight,
            *collapse_pre,
            *hc_scale,
            *hc_base,
            hc_norm_width,
            hc_norm_eps,
            hc_eps,
            hc_sinkhorn_iters,
            compute_kernel_config,
            comb_transposed,
            post_col,
            comb_col,
            output_tensor,
            stats_output,
            stats_scratch,
            hc_output,
            y_output,
            max_grid,
            norm_weight,
            norm_eps);
    }
    if (hc_output.has_value() || y_output.has_value() || hc_scale.has_value() || hc_base.has_value() ||
        norm_weight.has_value()) {
        throw ValueErrorCpp("mhc_post: hc_scale / hc_base / hc_output / y_output / norm_weight need collapse_pre");
    }
    if (stats_weight.has_value()) {
        return mhc_post_with_stats(
            input_tensor,
            residual,
            post,
            comb,
            *stats_weight,
            compute_kernel_config,
            comb_transposed,
            post_col,
            comb_col,
            output_tensor,
            stats_output,
            stats_scratch,
            max_grid);
    }
    if (stats_output.has_value() || stats_scratch.has_value()) {
        throw ValueErrorCpp("mhc_post: stats_output / stats_scratch need stats_weight");
    }
    return mhc_post(
        input_tensor,
        residual,
        post,
        comb,
        compute_kernel_config,
        comb_transposed,
        post_col,
        comb_col,
        output_tensor,
        max_grid);
}

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
