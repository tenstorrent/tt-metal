// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flip_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "flip.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_flip(py::module& module) {
    auto doc =
        R"doc(flip(input_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[MemoryConfig] = std::nullopt) -> ttnn.Tensor

            Reverse the order of an n-D tensor along given axis in dims.

            Input Specs:
                - **Supported:**
                    - Tensors with up to 4 dimensions
                    - Interleaved row-major layout tensors of following dtypes: `bfloat16`, `float32`, `int32`
                    - Interleaved tiled layout tensors of following dtypes: `bfloat16`, `float32`

                - **Not Supported:**
                    - Sharded tensors
                    - Interleaved tiled layout tensors  of `int32` dtype

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                dim (number): tthe permutation of the dimensions of the input tensor.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
               List of ttnn.Tensor: the output tensor.

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
                         [ 0,  1]]]))doc";

    using OperationType = decltype(ttnn::flip);
    ttnn::bind_registered_operation(
        module,
        ttnn::flip,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int64_t>& dims,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(input_tensor, dims, memory_config);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dims"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
        });
}

}  // namespace ttnn::operations::data_movement::detail
