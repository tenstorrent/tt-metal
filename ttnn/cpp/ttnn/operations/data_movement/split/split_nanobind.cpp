// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/decorators.hpp"

#include "split.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_split(nb::module_& mod) {
    auto doc =
        R"doc(
            Returns a tensor that is in num_splits ways on dim.

            Equivalent pytorch code:

            .. code-block:: python

                output_tensor = torch.split(input_tensor, 2, 1)

            Args:
                * :attr:`input_tensor`: Input Tensor.
                * :attr:`split_size` (Union[int, list[int]]): Single chunk size or list of chunk sizes. Output may be smaller if dim not evenly divisible.
                * :attr:`dim2`: Dim to split. Defaults to 0.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor
        )doc";

    using OperationType = decltype(ttnn::split);
    ttnn::bind_registered_operation(
        mod,
        ttnn::split,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int64_t split_size,
               const int64_t dim,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(input_tensor, split_size, dim, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("split_size"),
            nb::arg("dim") = 0,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()},
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int64_t>& split_sizes,
               const int64_t dim,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(input_tensor, split_sizes, dim, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("split_size"),
            nb::arg("dim") = 0,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}
}  // namespace ttnn::operations::data_movement::detail
