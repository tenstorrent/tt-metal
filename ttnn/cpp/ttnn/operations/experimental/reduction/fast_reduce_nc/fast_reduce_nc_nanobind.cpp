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
               nb::object dims_obj,
               const std::optional<const Tensor>& output,
               const ttnn::MemoryConfig& memory_config,
               const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
                ttnn::SmallVector<int32_t> dims;
                if (!dims_obj.is_none()) {
                    if (nb::isinstance<nb::int_>(dims_obj)) {
                        dims.push_back(nb::cast<int32_t>(dims_obj));
                    } else {
                        for (nb::handle h : nb::iter(dims_obj)) {
                            dims.push_back(nb::cast<int32_t>(h));
                        }
                    }
                }
                return self(input, dims, output, memory_config, compute_kernel_config);
            },
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("dims") = nb::none(),
            nb::arg("output") = nb::none(),
            nb::arg("memory_config") = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            nb::arg("compute_kernel_config") = nb::none()});
}

}  // namespace ttnn::operations::experimental::reduction::detail
