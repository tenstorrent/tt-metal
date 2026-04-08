// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "convert_to_chw.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::cnn::detail {

void bind_convert_to_chw(nb::module_& mod) {
    const auto* const doc = R"doc(
    Convert a tensor from HWC channel ordering to CHW channel ordering.

    The input tensor is expected to be tiled and height-sharded in L1. The output is a row-major width-sharded tensor.

    The output memory configuration is automatically inferred to create a width-sharded output
    with appropriate shard dimensions based on the input tensor's sharding configuration.
    )doc";

    ttnn::bind_function<"convert_to_chw", "ttnn.experimental.">(
        mod, doc, &ttnn::experimental::convert_to_chw, nb::arg("input"), nb::kw_only(), nb::arg("dtype") = nb::none());
}

}  // namespace ttnn::operations::experimental::cnn::detail
