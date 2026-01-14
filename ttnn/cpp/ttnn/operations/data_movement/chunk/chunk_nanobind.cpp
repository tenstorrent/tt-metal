// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_nanobind.hpp"

#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/data_movement/chunk/chunk.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace {

template <typename data_movement_operation_t>
void bind_chunk(nb::module_& mod, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod, operation, doc, ttnn::nanobind_arguments_t{nb::arg("input_tensor"), nb::arg("chunks"), nb::arg("dim")});
}

}  // namespace

void bind_chunk(nb::module_& mod) {
    bind_chunk(
        mod,
        ttnn::chunk,
        R"doc(
        Splits a tensor into multiple chunks along a specified dimension.

        Args:
            * :attr:`input_tensor`: The tensor to split.
            * :attr:`chunks`: Number of chunks to divide the tensor into.
            * :attr:`dim`: Dimension along which to split.

        )doc");
}

}  // namespace ttnn::operations::data_movement
