// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fast_reduce_nc_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <ttnn-nanobind/small_vector_caster.hpp>
#include "ttnn-nanobind/decorators.hpp"

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_fast_reduce_nc(nb::module_& mod) {
    using OperationType = decltype(ttnn::experimental::reduction::fast_reduce_nc);
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::reduction::fast_reduce_nc,
        R"doc(
              Performs optimized reduction operation on dim 0, 1, or [0,1]. Returns an output tensor.
        )doc",
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input,
               const ttnn::SmallVector<int32_t>& dims,
               const std::optional<const Tensor>& output,
               const ttnn::MemoryConfig& memory_config,
               std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
                return self(input, dims, output, memory_config, compute_kernel_config);
            },
            nb::arg("input").noconvert(),
            nb::kw_only(),
            nb::arg("dims").noconvert() = ttnn::SmallVector<int32_t>(),
            nb::arg("output").noconvert() = nb::none(),
            nb::arg("memory_config").noconvert() = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            nb::arg("compute_kernel_config").noconvert() = nb::none()});
}

}  // namespace ttnn::operations::experimental::reduction::detail
