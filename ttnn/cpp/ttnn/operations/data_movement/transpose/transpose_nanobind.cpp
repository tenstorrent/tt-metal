// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "transpose_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "transpose.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_transpose(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Returns a tensor that is transposed along dims dim1 and dim2

            Equivalent pytorch code:

            .. code-block:: python

                output_tensor = torch.transpose(input_tensor, 0, 1)

            Args:
                * :attr:`input_tensor`: Input Tensor.
                * :attr:`dim1`: First dim of transpose.
                * :attr:`dim2`: Second dim of transpose.
                * :attr:`pad_value` (float, optional): padding value for when tiles are broken in a transpose. Defaults to `0.0`.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor
        )doc";

    using OperationType = decltype(ttnn::transpose);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transpose,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               int64_t dim1,
               int64_t dim2,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               float pad_value = 0.0f) { return self(input_tensor, dim1, dim2, memory_config, pad_value); },
            nb::arg("input_tensor"),
            nb::arg("dim1"),
            nb::arg("dim2"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = 0.0f,
        });
}
}  // namespace ttnn::operations::data_movement::detail
