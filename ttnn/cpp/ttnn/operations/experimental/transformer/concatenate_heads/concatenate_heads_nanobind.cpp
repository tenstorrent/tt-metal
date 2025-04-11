// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "ttnn/operations/experimental/transformer/concatenate_heads/concatenate_heads.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_concatenate_heads(nb::module_& mod) {
    auto concatenate_heads_doc =
        R"doc(concatenate_heads(input_tensor: ttnn.Tensor, compute_with_storage_grid_size: ttnn.CoreCoord: *, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

            Reshuffles a [9, 16, 384, 64] ttnn.Layout.TILE BFLOAT8_B tensor into a tensor with shape [9, 1, 384, 1024].

            Args:
                * :attr:`input_tensor`: Input Tensor with shape [9, 16, 384, 64], dtype BFLOAT8_B and layout ttnn.Layout.TILE
                * :attr:`compute_with_storage_grid_size`: Compute Grid

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to input_tensor.memory_config()
        )doc";

    using OperationType = decltype(ttnn::experimental::concatenate_heads);

    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::concatenate_heads,
        concatenate_heads_doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const CoreCoord& compute_with_storage_grid_size,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> optional_output_tensor) {
                return self(input_tensor, compute_with_storage_grid_size, memory_config, optional_output_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("compute_with_storage_grid_size").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

}  // namespace ttnn::operations::experimental::transformer::detail
