// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "deepseek_moe_fast_reduce_nc_nanobind.hpp"
#include "ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/deepseek_moe_fast_reduce_nc.hpp"

#include <ttnn-nanobind/small_vector_caster.hpp>
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_deepseek_moe_fast_reduce_nc(nb::module_& mod) {
    using OperationType = decltype(ttnn::experimental::reduction::deepseek_moe_fast_reduce_nc);
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::reduction::deepseek_moe_fast_reduce_nc,
        R"doc(
        Performs optimized reduction operation on tensor of at least rank 3, reducing on any of the dims but the last 2

        Args:
            input_tensor (ttnn.Tensor): the input tensor
            dim (int): dimension along which to reduce

        Keyword Args:
            split_size (int): size of last dim of each output split tensor
            output_memory_config (ttnn.MemoryConfig): output memory configuration
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): configuration for the reduction

        Returns:
            list of ttnn.Tensor: returns x split tensors, representative of the single logical output tensor split on the last dim
        )doc",
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               int32_t dim,
               uint64_t split_size,
               const tt::tt_metal::MemoryConfig& output_memory_config,
               const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
                return self(input_tensor, dim, split_size, output_memory_config, compute_kernel_config);
            },
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("split_size"),
            nb::arg("output_memory_config").noconvert() = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            nb::arg("compute_kernel_config").noconvert() = nb::none()});
}

}  // namespace ttnn::operations::experimental::reduction::detail
