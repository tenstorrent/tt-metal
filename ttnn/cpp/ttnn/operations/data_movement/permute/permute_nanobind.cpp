// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "permute.hpp"
#include "permute_force.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_permute(nb::module_& mod) {
    const auto* doc = R"doc(
        Permutes the dimensions of the input tensor according to the specified permutation.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dim (number): the permutation of the dimensions of the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            pad_value (float, optional): padding value for when tiles are broken in a transpose. Defaults to `0.0`.

        Returns:
            List of ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"permute">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttsl::SmallVector<int64_t>&,
                const std::optional<ttnn::MemoryConfig>&,
                float>(&ttnn::permute),
            nb::arg("input_tensor").noconvert(),
            nb::arg("dims"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = 0.0f),
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, const ttsl::SmallVector<int64_t>&, float>(&ttnn::permute),
            nb::arg("input_tensor").noconvert(),
            nb::arg("dims"),
            nb::arg("pad_value") = 0.0f));

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See permute_force.hpp.
    mod.def(
        "permute_force_native",
        &permute_force_native,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dims"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("pad_value") = 0.0f,
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native permute implementation unconditionally. Not part of
            the ttnn API; use ttnn.permute, which selects an implementation on its own.
        )doc");

    mod.def(
        "permute_force_codegen",
        &permute_force_codegen,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dims"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("pad_value") = 0.0f,
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the codegen permute implementation unconditionally, raising for
            a case outside its support scope rather than falling back to native. Not part of the
            ttnn API; use ttnn.permute, which selects an implementation on its own.
        )doc");
}

}  // namespace ttnn::operations::data_movement::detail
