// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_nanobind.hpp"

#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/chunk/chunk.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_chunk(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Splits a tensor into multiple chunks along a specified dimension.

        Args:
            * :attr:`input_tensor`: The tensor to split.
            * :attr:`chunks`: Number of chunks to divide the tensor into.
            * :attr:`dim`: Dimension along which to split.

        )doc";

    ttnn::bind_function<"chunk">(
        mod, doc, ttnn::overload_t(&ttnn::chunk, nb::arg("input_tensor"), nb::arg("chunks"), nb::arg("dim")));
}

}  // namespace ttnn::operations::data_movement
