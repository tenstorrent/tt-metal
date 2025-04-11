// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "full_like_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "full_like.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/full_like/device/full_like_device_operation.hpp"

namespace ttnn::operations::full_like {

void bind_full_like_operation(nb::module_& mod) {
    auto doc =
        R"doc(full_like(tensor: Tensor, fill_value: float or value, dtype: DataType, layout: Layout, memory_config: MemoryConfig) -> Tensor

    Create a tensor with the same shape of the given tensor and filled with given fill_value, with the specified `memory_config` and converting its data type to `dtype`.
    This operation only supports TILE_LAYOUT for now.

    Args:
        * :attr:`input`: The tensor has shape which will be based on to make the output tensor
        * :attr:`fill_value`: The value which will be used to fill the output tensor
        * :attr:`dtype`: The target data type of the output tensor.
        * :attr:`layout`: The target layout of the output tensor.
        * :attr:`memory_config`: The memory configuration for the output tensor.
    )doc";

    bind_registered_operation(
        mod,
        ttnn::moreh_full_like,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("fill_value"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("layout") = nb::none(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::full_like
