// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "deepseek_moe_post_combine_tilize_nanobind.hpp"
#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/deepseek_moe_post_combine_tilize.hpp"

#include <ttnn-nanobind/small_vector_caster.hpp>
#include "ttnn-nanobind/decorators.hpp"

// namespace ttnn::operations::experimental::deepseek_moe_post_combine_tilize::detail {

// void bind_deepseek_moe_post_combine_tilize(nb::module_& mod) {
//     using OperationType = decltype(ttnn::experimental::deepseek_moe_post_combine_tilize);
//     ttnn::bind_registered_operation(
//         mod,
//         ttnn::experimental::deepseek_moe_post_combine_tilize,
//         // TODO: (GR)
//         R"doc(
//         Performs optimized reduction operation on tensor of at least rank 3, reducing on any of the dims but the last
//         2

//         Args:
//             input_tensor (ttnn.Tensor): the input tensor
//             dim (int): dimension along which to reduce

//         Keyword Args:
//             split_size (int): size of last dim of each output split tensor
//             output_memory_config (ttnn.MemoryConfig): output memory configuration
//             compute_kernel_config (ttnn.DeviceComputeKernelConfig): configuration for the reduction

//         Returns:
//             list of ttnn.Tensor: returns x split tensors, representative of the single logical output tensor split on
//             the last dim
//         )doc",
//         ttnn::nanobind_overload_t{
//             [](const OperationType& self,
//                const ttnn::Tensor& input_tensor,
//                const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
//                 return self(input_tensor, output_memory_config);
//             },
//             nb::arg("input_tensor"),
//             nb::kw_only(),
//             nb::arg("output_memory_config").noconvert() = nb::none()});
// }

// }  // namespace ttnn::operations::experimental::deepseek_moe_post_combine_tilize::detail

namespace ttnn::operations::experimental::deepseek_moe_post_combine_tilize::detail {
namespace nb = nanobind;

void bind_deepseek_moe_post_combine_tilize(nb::module_& mod) {
    // TODO: (GR)
    const auto* doc =
        R"doc(
            Writes the input tensor to a slice of the output tensor.

            Constraints:
                Input & Output must have rank == 4.
                DType must be bfloat16.
                Supports only Row Major Tensors.
                Output Tensor must be interleaved.
                Input Tensor can be interleaved, height sharded or block sharded.
                Steps must be all ones.
                Slicing along the last dimension is not supported.

            Args:
                input_tensor: Input Tensor.
                output_tensor: Output Tensor.
                slice_start: Start indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                slice_end: End indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                slice_step: (Optional[List[int[tensor rank]]) Step size for each dim. Default is None, which works out be 1 for each dimension.

            Keyword Args:
                memory_config Memory Config of the output tensor

            Returns:
                ttnn.Tensor: the output tensor after writing the input tensor to it.

            Example:
                >>> ttnn.experimental.slice_write(ttnn_input_tensor, ttnn_output_tensor, output_start_indices, output_end_indices, strides)
                )doc";

    ttnn::bind_function<"deepseek_moe_post_combine_tilize", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::deepseek_moe_post_combine_tilize,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("output_memory_config").noconvert() = nb::none());
}
}  // namespace ttnn::operations::experimental::deepseek_moe_post_combine_tilize::detail
