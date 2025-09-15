// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_pybind.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_fast_reduce_nc(pybind11::module& module) {
    using OperationType = decltype(ttnn::experimental::reduction::fast_reduce_nc);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::reduction::fast_reduce_nc,
        R"doc(
              Performs optimized reduction operation on dim 0, 1, or [0,1]. Returns an output tensor.
        )doc",
        ttnn::pybind_arguments_t{
            pybind11::arg("input").noconvert(),
            pybind11::kw_only(),
            pybind11::arg("dims").noconvert() = ttnn::SmallVector<int32_t>(),
            pybind11::arg("output").noconvert() = std::nullopt,
            pybind11::arg("memory_config").noconvert() = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            pybind11::arg("compute_kernel_config").noconvert() = std::nullopt});
}

}  // namespace ttnn::operations::experimental::reduction::detail
