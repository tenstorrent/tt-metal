// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "slice_write.hpp"

namespace ttnn::operations::experimental::quasar::detail {
namespace nb = nanobind;

void bind_slice_write(nb::module_& mod) {
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
                start: Start indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                end: End indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                step: (List[int[tensor rank]]) Step size for each dim. Required; must be all ones.

            Returns:
                ttnn.Tensor: the output tensor after writing the input tensor to it.

            Example:
                >>> input_tensor = ttnn.from_torch(torch.rand(1, 4, 4, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                >>> output_tensor = ttnn.from_torch(torch.zeros(1, 8, 8, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                >>> ttnn.experimental.slice_write(input_tensor, output_tensor, [0, 0, 0, 0], [1, 4, 4, 32], [1, 1, 1, 1])
                )doc";

    // TODO: implementing the array version and overloading the nanobind with all the possible array sizes is better
    // than a vector with a fixed size default value
    ttnn::bind_function<"slice_write", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::slice_write,
        nb::arg("input_tensor"),
        nb::arg("output_tensor"),
        nb::arg("start"),
        nb::arg("end"),
        nb::arg("step"));
}
}  // namespace ttnn::operations::experimental::quasar::detail
