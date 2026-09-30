// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_ttnn_nanobind.hpp"

#include <exception>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_post_ttnn.hpp"
#include "device/mhc_post_ttnn_device_operation_types.hpp"
#include "device/mhc_post_ttnn_program_factory.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn::detail {

namespace {

tt::tt_metal::ProgramDescriptor program_descriptor_py(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    const ttnn::Tensor& output_tensor,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    bool comb_transposed) {
    return create_program_descriptor(
        input_tensor,
        residual,
        post,
        comb,
        output_tensor,
        compute_kernel_config.value_or(default_compute_kernel_config()),
        comb_transposed);
}

}  // namespace

void bind_mhc_post_ttnn(nb::module_& mod) {
    // The refusals keep the Python op's exception types (see mhc_post_ttnn.cpp).
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

    ttnn::bind_function<"mhc_post", "ttnn.bringup.">(
        mod,
        R"doc(
        mHC post-sublayer mix in one device program -- the AI-generated, perf-optimized op of
        ttnn/ttnn/bringup/mhc_post_ttnn. C++ host side; the device kernels are the op's.

            X'[t, j*C:(j+1)*C] = post[t, j] * F[t, :] + sum_i comb[t, i*n + j] * X[t, i*C:(i+1)*C]

        (comb applied transposed).

        Args:
            input_tensor (ttnn.Tensor): F, the sublayer output (..., T, C), TILE, DRAM interleaved, C % 32 == 0.
            residual (ttnn.Tensor): X, the n residual streams (..., T, n*C), float32 or bfloat16.
            post (ttnn.Tensor): (..., T, n), float32 TILE, 1 <= n <= 5.
            comb (ttnn.Tensor): (..., T, n*n), float32 TILE, comb[..., i*n + j] = comb[i][j].

        Keyword Args:
            compute_kernel_config (ttnn.ComputeConfigDescriptor, optional): fp32_dest_acc_en must be True.
                Defaults to HiFi4, fp32 DEST, approx off.
            comb_transposed (bool, optional): True (default) applies comb transposed as above. False applies it
                as stored: X'[t, j*C:(j+1)*C] = post[t, j] * F[t, :] + sum_i comb[t, j*n + i] * X[t, i*C:(i+1)*C].

        Returns:
            ttnn.Tensor: X' (..., T, n*C), the residual's dtype, TILE, DRAM interleaved.
        )doc",
        &mhc_post,
        nb::arg("input_tensor"),
        nb::arg("residual"),
        nb::arg("post"),
        nb::arg("comb"),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("comb_transposed") = true);

    mod.def(
        "_mhc_post_ttnn_program_descriptor",
        &program_descriptor_py,
        nb::arg("input_tensor"),
        nb::arg("residual"),
        nb::arg("post"),
        nb::arg("comb"),
        nb::arg("output_tensor"),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("comb_transposed") = true,
        R"doc(Build ttnn.bringup.mhc_post's ProgramDescriptor with the C++ builder (no dispatch). Test-only.)doc");
}

}  // namespace ttnn::operations::bringup::mhc_post_ttnn::detail
