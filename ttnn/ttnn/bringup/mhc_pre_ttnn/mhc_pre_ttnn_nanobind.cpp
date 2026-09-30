// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_ttnn_nanobind.hpp"

#include <exception>
#include <optional>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_pre_ttnn.hpp"
#include "device/mhc_pre_ttnn_device_operation_types.hpp"
#include "device/mhc_pre_ttnn_program_factory.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn::detail {

namespace {

tt::tt_metal::ProgramDescriptor program_descriptor_py(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& proj_weight,
    const ttnn::Tensor& proj_bias,
    const ttnn::Tensor& y,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    uint32_t n,
    const std::vector<double>& scale,
    uint32_t sinkhorn_iters,
    double eps,
    double norm_eps,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config) {
    TT_FATAL(scale.size() == 3, "scale must hold 3 values");
    const MhcPreParams params{
        .n = n,
        .scale = {scale[0], scale[1], scale[2]},
        .sinkhorn_iters = sinkhorn_iters,
        .eps = eps,
        .norm_eps = norm_eps,
        .compute_config = compute_kernel_config.value_or(default_compute_kernel_config())};
    return create_program_descriptor(input_tensor, proj_weight, proj_bias, y, post, comb, params);
}

}  // namespace

void bind_mhc_pre_ttnn(nb::module_& mod) {
    // The refusals keep the Python op's exception types (see mhc_pre_ttnn.cpp).
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void*) {
            try {
                std::rethrow_exception(p);
            } catch (const UnsupportedAxisError& e) {
                PyObject* cls = PyExc_NotImplementedError;
                nb::object contract;
                try {
                    contract = nb::module_::import_("ttnn.operations._op_contract").attr("UnsupportedAxisValue");
                    cls = contract.ptr();
                } catch (...) {
                    PyErr_Clear();
                }
                PyErr_SetString(cls, e.what());
            }
        },
        nullptr);

    ttnn::bind_function<"mhc_pre", "ttnn.bringup.">(
        mod,
        R"doc(
        mHC pre-sublayer half in one device program -- the AI-generated, perf-optimized op of
        ttnn/ttnn/bringup/mhc_pre_ttnn. C++ host side; the device kernels are the op's.

        Per token row x (length n*C) of X (..., T, n*C):
            r     = rsqrt(mean(x^2) + norm_eps)
            mixes = (x @ W) * r
            pre   = sigmoid(a_pre  * mixes[0:n]  + b[0:n]) + eps
            post  = 2 * sigmoid(a_post * mixes[n:2n] + b[n:2n])
            comb  = Sinkhorn(a_res * mixes[2n:] + b[2n:])        (n x n, DeepSeek-V4 order)
            y     = sum_i pre[i] * x[i*C:(i+1)*C]

        Args:
            input_tensor (ttnn.Tensor): X (..., T, n*C), float32 or bfloat16, TILE, C % 32 == 0.
            proj_weight (ttnn.Tensor): W (n*C, n (n + 2)), float32 or bfloat16, TILE.
            proj_bias (ttnn.Tensor): b (1, n (n + 2)), float32 TILE.

        Keyword Args:
            scale (tuple[float, float, float]): (a_pre, a_post, a_res).
            sinkhorn_iters (int): >= 1. Defaults to 20.
            eps (float): Defaults to 1e-6.
            norm_eps (float): Defaults to 1e-6.
            compute_kernel_config (ttnn.ComputeConfigDescriptor, optional): fp32_dest_acc_en must be True.
                Defaults to HiFi4, fp32 DEST, approx off.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: y (..., T, C) in X's dtype, post (..., T, n) and
            comb (..., T, n*n) in float32; TILE, DRAM interleaved.
        )doc",
        &mhc_pre,
        nb::arg("input_tensor"),
        nb::arg("proj_weight"),
        nb::arg("proj_bias"),
        nb::kw_only(),
        nb::arg("scale"),
        nb::arg("sinkhorn_iters") = 20,
        nb::arg("eps") = 1e-6,
        nb::arg("norm_eps") = 1e-6,
        nb::arg("compute_kernel_config") = nb::none());

    mod.def(
        "_mhc_pre_ttnn_program_descriptor",
        &program_descriptor_py,
        nb::arg("input_tensor"),
        nb::arg("proj_weight"),
        nb::arg("proj_bias"),
        nb::arg("y"),
        nb::arg("post"),
        nb::arg("comb"),
        nb::kw_only(),
        nb::arg("n"),
        nb::arg("scale"),
        nb::arg("sinkhorn_iters"),
        nb::arg("eps"),
        nb::arg("norm_eps"),
        nb::arg("compute_kernel_config") = nb::none(),
        R"doc(Build ttnn.bringup.mhc_pre's ProgramDescriptor with the C++ builder (no dispatch). Test-only.)doc");
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn::detail
