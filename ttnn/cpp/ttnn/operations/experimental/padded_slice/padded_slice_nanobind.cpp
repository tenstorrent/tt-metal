// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "padded_slice.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::padded_slice {

void bind_padded_slice(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Returns a padded_sliced tensor. If the input tensor is on host, the padded_slice will be performed on host, and if its on device it will be performed on device.

            Args:
                input_tensor: Input Tensor.
                padded_slice_start: Start indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                padded_slice_end: End indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                padded_slice_step: (Optional[List[int[tensor rank]]) Step size for each dim. Default is None, which works out be 1 for each dimension.
                memory_config: Memory Config of the output tensor. This must be either height or block sharded.

            Returns:
                ttnn.Tensor: the output tensor.

            Example:
                >>> tensor = ttnn.padded_slice(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device), [0, 0, 0, 0], [1, 1, 64, 16], [1, 1, 2, 1])
                >>> print(tensor.shape)
                [1, 1, 32, 16]
                >>> input = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.padded_slice(input, [0, 0, 0, 0], [1, 1, 32, 32])
                >>> print(output.shape)
                [1, 1, 32, 32]
                )doc";

    // TODO: implementing the array version and overloading the nanobind with all the possible array sizes is better
    // than a vector with a fixed size default value
    using OperationType = decltype(ttnn::experimental::padded_slice);
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::padded_slice,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int>& padded_slice_start,
               const ttnn::SmallVector<int>& padded_slice_end,
               const std::optional<ttnn::SmallVector<int>>& step,
               const MemoryConfig& memory_config,
               const std::optional<Tensor>& optional_output_tensor,
               const std::optional<float>& /*pad_value*/) {
                const auto step_value = step.value_or(ttnn::SmallVector<int>(padded_slice_end.size(), 1));
                return self(
                    input_tensor,
                    padded_slice_start,
                    padded_slice_end,
                    step_value,
                    memory_config,
                    optional_output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("padded_slice_start"),
            nb::arg("padded_slice_end"),
            nb::arg("padded_slice_step") = nb::none(),  // should consider a better default value
            nb::kw_only(),
            nb::arg("memory_config"),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none()});
}
}  // namespace ttnn::operations::experimental::padded_slice
