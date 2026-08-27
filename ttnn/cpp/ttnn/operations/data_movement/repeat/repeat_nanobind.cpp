// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"  // for ttsl::SmallVector<uint32_t>

#include "repeat.hpp"
#include "repeat_force.hpp"

namespace ttnn::operations::data_movement {
namespace nb = nanobind;

void bind_repeat(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a new tensor filled with repetition of input :attr:`input_tensor` according to number of times specified in :attr:`shape`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            repetition_vector (SmallVector): The number of repetitions for each dimension.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"repeat">(
        mod,
        doc,
        nb::overload_cast<
            const ttnn::Tensor&,
            const ttsl::SmallVector<uint32_t>&,
            const std::optional<MemoryConfig>&,
            const std::optional<ttnn::Tensor>&>(&ttnn::repeat),
        nb::arg("input_tensor"),
        nb::arg("repeat_dims"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("optional_output_tensor") = nb::none());

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See repeat_force.hpp.
    mod.def(
        "repeat_force_native",
        &detail::repeat_force_native,
        nb::arg("input_tensor"),
        nb::arg("repeat_dims"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native repeat implementation unconditionally. Not part of the
            ttnn API; use ttnn.repeat, which selects an implementation on its own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                repeat_dims (SmallVector): the number of repetitions for each dimension.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.

            Returns:
                ttnn.Tensor: the output tensor. Takes no preallocated output -- a forced leg is a
                reference for comparison, not a call path with the full public surface.
        )doc");

    mod.def(
        "repeat_force_codegen",
        &detail::repeat_force_codegen,
        nb::arg("input_tensor"),
        nb::arg("repeat_dims"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the codegen repeat implementation unconditionally, raising for a
            case outside its support scope rather than falling back to native. Not part of the ttnn
            API; use ttnn.repeat, which selects an implementation on its own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                repeat_dims (SmallVector): the number of repetitions for each dimension.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.

            Returns:
                ttnn.Tensor: the output tensor. Takes no preallocated output -- a forced leg is a
                reference for comparison, not a call path with the full public surface.

            Raises:
                RuntimeError: the codegen path does not support this case.
        )doc");
}

}  // namespace ttnn::operations::data_movement
