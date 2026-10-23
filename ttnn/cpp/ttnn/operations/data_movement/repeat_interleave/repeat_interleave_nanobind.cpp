// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_interleave_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "repeat_interleave.hpp"
#include "repeat_interleave_force.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_repeat_interleave(nb::module_& mod) {
    const auto* doc = R"doc(
        Repeats elements of a :attr:`tensor` in the given :attr:`dim`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            repeats (number): he number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
            dim (number): the dimension to expand with the repetitions.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"repeat_interleave">(
        mod,
        doc,
        &ttnn::repeat_interleave,
        nb::arg("input_tensor"),
        nb::arg("repeats"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    // Bound with a plain def rather than ttnn::bind_function: the latter tags the callable for
    // auto_register_ttnn_cpp_operations, which would republish these as ttnn.* operations. They are
    // meant to stay reachable only via this private module. See repeat_interleave_force.hpp.
    mod.def(
        "repeat_interleave_force_native",
        &repeat_interleave_force_native,
        nb::arg("input_tensor"),
        nb::arg("repeats"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the native repeat_interleave implementation unconditionally. Not
            part of the ttnn API; use ttnn.repeat_interleave, which selects an implementation on its
            own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                repeats (number): the number of repetitions for each element.
                dim (number): the dimension to expand with the repetitions.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.

            Returns:
                ttnn.Tensor: the output tensor.
        )doc");

    mod.def(
        "repeat_interleave_force_codegen",
        &repeat_interleave_force_codegen,
        nb::arg("input_tensor"),
        nb::arg("repeats"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
            Verification only: runs the codegen repeat_interleave implementation unconditionally,
            raising for a case outside its support scope rather than falling back to native. Not
            part of the ttnn API; use ttnn.repeat_interleave, which selects an implementation on its
            own.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                repeats (number): the number of repetitions for each element.
                dim (number): the dimension to expand with the repetitions.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the output.
                    Defaults to `None`, which derives it from the input.

            Returns:
                ttnn.Tensor: the output tensor.

            Raises:
                RuntimeError: the codegen path does not support this case.
        )doc");
}

}  // namespace ttnn::operations::data_movement::detail
