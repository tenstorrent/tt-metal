// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_ttnn.hpp"

#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "device/mhc_pre_ttnn_device_operation.hpp"
#include "device/mhc_pre_ttnn_device_operation_types.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

namespace {

std::vector<int64_t> dims(const Tensor& t) {
    const auto& s = t.logical_shape();
    std::vector<int64_t> d;
    for (size_t i = 0; i < s.rank(); ++i) {
        d.push_back(s[i]);
    }
    return d;
}

std::string tuple_str(const std::vector<int64_t>& d) {
    return d.size() == 1 ? fmt::format("({},)", d[0]) : fmt::format("({})", fmt::join(d, ", "));
}

// SUPPORTED (mhc_pre.py): dtype / weight_dtype in {float32, bfloat16}, layout TILE, fp32_dest_acc_en True,
// alignment in {tile_aligned, h_non_aligned} (every T). EXCLUSIONS is empty.
void check_supported(const Tensor& x, const Tensor& w, const tt::tt_metal::ComputeConfigDescriptor& cfg) {
    auto float_types = [](DataType d) { return d == DataType::FLOAT32 || d == DataType::BFLOAT16; };
    if (!float_types(x.dtype())) {
        throw UnsupportedAxisError(
            fmt::format("mhc_pre: dtype={} not in SUPPORTED [DataType.FLOAT32, DataType.BFLOAT16]", x.dtype()));
    }
    if (x.layout() != Layout::TILE) {
        throw UnsupportedAxisError("mhc_pre: layout=ROW_MAJOR not in SUPPORTED [Layout.TILE]");
    }
    if (!float_types(w.dtype())) {
        throw UnsupportedAxisError(
            fmt::format("mhc_pre: weight_dtype={} not in SUPPORTED [DataType.FLOAT32, DataType.BFLOAT16]", w.dtype()));
    }
    if (!cfg.fp32_dest_acc_en) {
        throw UnsupportedAxisError("mhc_pre: fp32_dest_acc_en=False not in SUPPORTED [True]");
    }
}

int64_t derive_n(int64_t mix) {
    for (int64_t n = 1; n < 64; ++n) {
        if (n * (n + 2) == mix) {
            return n;
        }
        if (n * (n + 2) > mix) {
            break;
        }
    }
    throw ValueErrorCpp(fmt::format("mhc_pre: proj_weight last dim {} is not n*(n+2) for any integer n", mix));
}

int64_t check_contract(const Tensor& x, const Tensor& w, const Tensor& b, int64_t sinkhorn_iters) {
    const auto xs = dims(x), ws = dims(w), bs = dims(b);
    if (xs.size() < 2 || xs.size() > 4) {
        throw ValueErrorCpp(fmt::format("mhc_pre: input rank must be 2..4, got {}", xs.size()));
    }
    if (ws.size() != 2) {
        throw ValueErrorCpp(fmt::format("mhc_pre: proj_weight must be 2D, got {}", tuple_str(ws)));
    }
    const int64_t n = derive_n(ws.back());
    if (n * (n + 2) + 1 > 32) {
        throw ValueErrorCpp(fmt::format("mhc_pre: n={} needs {} coefficient slots (> 32)", n, n * (n + 2) + 1));
    }
    const int64_t nc = xs.back();
    if (nc % n != 0 || (nc / n) % 32 != 0) {
        throw ValueErrorCpp(fmt::format("mhc_pre: last dim {} must be n*C with C % 32 == 0 (n={})", nc, n));
    }
    if (ws[0] != nc) {
        throw ValueErrorCpp(
            fmt::format("mhc_pre: proj_weight shape {} does not match input last dim {}", tuple_str(ws), nc));
    }
    if (bs.back() != n * (n + 2) || b.dtype() != DataType::FLOAT32 || b.layout() != Layout::TILE) {
        throw ValueErrorCpp(fmt::format(
            "mhc_pre: proj_bias must be float32 TILE of shape (1, {}), got {}", n * (n + 2), tuple_str(bs)));
    }
    if (w.layout() != Layout::TILE) {
        throw ValueErrorCpp("mhc_pre: proj_weight must be TILE_LAYOUT");
    }
    if (sinkhorn_iters < 1) {
        throw ValueErrorCpp(fmt::format("mhc_pre: sinkhorn_iters must be >= 1, got {}", sinkhorn_iters));
    }
    return n;
}

}  // namespace

tt::tt_metal::ComputeConfigDescriptor default_compute_kernel_config() {
    tt::tt_metal::ComputeConfigDescriptor c;
    c.math_fidelity = MathFidelity::HiFi4;
    c.fp32_dest_acc_en = true;
    c.math_approx_mode = false;
    return c;
}

std::tuple<Tensor, Tensor, Tensor> mhc_pre(
    const Tensor& input_tensor,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const std::vector<double>& scale,
    int64_t sinkhorn_iters,
    double eps,
    double norm_eps,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config) {
    const auto cfg = compute_kernel_config.value_or(default_compute_kernel_config());
    check_supported(input_tensor, proj_weight, cfg);
    const int64_t n = check_contract(input_tensor, proj_weight, proj_bias, sinkhorn_iters);
    if (scale.size() != 3) {
        throw ValueErrorCpp(fmt::format("mhc_pre: scale must be (a_pre, a_post, a_res), got {} values", scale.size()));
    }
    MhcPreParams params{
        .n = static_cast<uint32_t>(n),
        .scale = {scale[0], scale[1], scale[2]},
        .sinkhorn_iters = static_cast<uint32_t>(sinkhorn_iters),
        .eps = eps,
        .norm_eps = norm_eps,
        .compute_config = cfg};
    return ttnn::prim::bringup::mhc_pre_ttnn(input_tensor, proj_weight, proj_bias, params);
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
