// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <ttnn-nanobind/decorators.hpp>

#include "permute.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_permute(nb::module_& mod) {
    auto doc =
        R"doc(permute(input_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[MemoryConfig] = std::nullopt)) -> ttnn.Tensor

            Permutes the dimensions of the input tensor according to the specified permutation.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                dim (number): tthe permutation of the dimensions of the input tensor.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                pad_value (float, optional): padding value for when tiles are broken in a transpose. Defaults to `0.0`. If set to None, it will be random garbage values.

           Returns:
               List of ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.permute(tensor, (0, 1, 3, 2))
                >>> print(output.shape)
                [1, 1, 32, 64])doc";

    using OperationType = decltype(ttnn::permute);
    ttnn::bind_registered_operation(
        mod,
        ttnn::permute,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int64_t>& dims,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<float>& pad_value) { return self(input_tensor, dims, memory_config, pad_value); },
            nb::arg("input_tensor").noconvert(),
            nb::arg("dims"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = 0.0f,
        });
}

}  // namespace ttnn::operations::data_movement::detail
