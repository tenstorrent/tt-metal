// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_to_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/bcast_to/bcast_to.hpp"

namespace ttnn::operations::experimental::broadcast_to::detail {

void bind_broadcast_to(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Returns a new tensor where singleton dimensions are broadcasted to the given shape.

        Args:
            * :attr:`input`: The tensor to be broadcasted.
            * :attr:`output_shape`: The desired broadcasted shape.
            * :attr:`memory_config`: The memory configuration for the broadcasted tensor.
            * :attr:`output`: An optional tensor to store the broadcasted result.

        Notes:
            Currently Supports:
            Data Type
	            bfloat16
	            float32

            Tensor Shape
                up to 4D

            Memory Layout
                Tile

            Memory Config
                Interleaved (DRAM / L1)
        )doc";
    ttnn::bind_function<"broadcast_to", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::broadcast_to,
        nb::arg("input"),
        nb::arg("output_shape"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output") = nb::none());
}
}  // namespace ttnn::operations::experimental::broadcast_to::detail
