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
#include "mhc_pre_xing.hpp"
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

    ttnn::bind_function<"mhc_pre_xing", "ttnn.bringup.">(
        mod,
        R"doc(
        mHC coefficients + collapse after the caller's own projection all-reduce (Xing4.0 math; xing40_a4b_d_p
        P.2b). A separate entry of the mhc_pre fork: ttnn.bringup.mhc_pre is unchanged.

        coefficients_given=False: input is the all-reduced row (..., T, 32) fp32 = [mix 0..n(n+2)-1 | sum x^2 | 0..]
        (unnormalised x @ fn^T and sum x^2 over the full n*H width). Per token:
            r    = rsqrt(sum x^2 / norm_width + norm_eps);  z = scale_g * mix * r + base
            pre  = sigmoid(z[0:n])                    (no + eps)
            post = 2 * sigmoid(z[n:2n])
            L    = clamp(z[2n:], clamp_min, clamp_max)   (L[i][j] at i*n + j)
            comb = exp(L - rowmax); sinkhorn_iters x {m / (rowsum + hc_eps), then m / (colsum + hc_eps)}
        -> hc (..., T, n(n+2)) fp32 = [pre | post | comb row-major] (the layout tt/residual.py slices for
        mhc_post(comb_transposed=False)).
        coefficients_given=True: input is a finished hc (..., T, n(n+2)); only pre is read.
        streams (..., T, n*C) fp32, optional: y = sum_i pre_i * streams[..., i*C:(i+1)*C] -> (..., T, C) fp32.

        Returns:
            tuple[ttnn.Tensor | None, ttnn.Tensor | None]: (hc, y); hc when computed, y when streams are given.
        )doc",
        &mhc_pre_xing,
        nb::arg("input_tensor"),
        nb::arg("streams") = nb::none(),
        nb::kw_only(),
        nb::arg("scale") = std::vector<double>{},
        nb::arg("base") = std::vector<double>{},
        nb::arg("norm_width") = 1.0,
        nb::arg("n") = 4,
        nb::arg("norm_eps") = 1e-6,
        nb::arg("hc_eps") = 1e-6,
        nb::arg("sinkhorn_iters") = 20,
        nb::arg("clamp_min") = -30.0,
        nb::arg("clamp_max") = 30.0,
        nb::arg("coefficients_given") = false,
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"mhc_pre_xing_pack", "ttnn.bringup.">(
        mod,
        R"doc(
        The row ttnn.bringup.mhc_pre_xing reduces, from the caller's partial projection, in one pass over the streams.

        Args:
            mix (ttnn.Tensor): (..., T, 32) float32 TILE: the partial x @ fn^T in columns 0 .. n(n+2)-1, zero elsewhere.
            streams (ttnn.Tensor): (..., T, n*C) float32 TILE, the local stream columns.

        Returns:
            ttnn.Tensor: mix with column n (n + 2) = sum over the row of streams^2 (exact fp32 on the SFPU).
        )doc",
        &mhc_pre_xing_pack,
        nb::arg("mix"),
        nb::arg("streams"),
        nb::kw_only(),
        nb::arg("n") = 4);

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
