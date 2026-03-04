// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/flip/flip_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn/operations/data_movement/flip/flip.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_flip_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::flip,
        R"doc(
        Reverse the order of an n-D tensor along given axis in dims.

        Input Specs:
            - **Supported:**
                - Interleaved row-major layout tensors of following dtypes: `bfloat16`, `float32`, `int32`
                - Interleaved tiled layout tensors of following dtypes: `bfloat16`, `float32`, `int32`

            - **Not Supported:**
                - Sharded tensors

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dims (List[int]): the dimensions to reverse.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> # Create a simple tensor
            >>> torch_tensor = torch.arange(12).reshape(3, 4).float()
            >>> tensor = ttnn.from_torch(torch_tensor, device=device)
            >>>
            >>> # Flip along dimension 0 (rows)
            >>> flipped = ttnn.flip(tensor, dims=[0])
            >>>
            >>> # Flip along dimension 1 (columns)
            >>> flipped_cols = ttnn.flip(tensor, dims=[1])
            >>>
            >>> # Flip along both dimensions
            >>> flipped_both = ttnn.flip(tensor, dims=[0, 1])

            >>> x = ttnn.to_device(ttnn.from_torch(torch.arrange(8).view(2, 2, 2), dtype=torch.bfloat16)), device)
            >>> x
            tensor([[[ 0,  1],
                     [ 2,  3]],

                    [[ 4,  5],
                     [ 6,  7]]])
            >>> flipped_x = ttnn.flip(x, (0, 1))
            >>> flipped_x
            tensor([[[ 6,  7],
                     [ 4,  5]],

                    [[ 2,  3],
                     [ 0,  1]]])
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::flip)& op,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int64_t>& dims,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> ttnn::Tensor {
                return op(input_tensor, dims, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("dims"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::data_movement
