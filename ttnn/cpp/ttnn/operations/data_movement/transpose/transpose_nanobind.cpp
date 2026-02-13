// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "transpose_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

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

    // Bind the free functions directly
    ttnn::bind_function<"transpose">(
        mod,
        doc,

        // Overload 1: with memory_config
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int64_t, int64_t, const std::optional<ttnn::MemoryConfig>&, float>(
                &ttnn::transpose),
            nb::arg("input_tensor"),
            nb::arg("dim1"),
            nb::arg("dim2"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = 0.0f),

        // Overload 2: without memory_config
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int64_t, int64_t, float>(&ttnn::transpose),
            nb::arg("input_tensor"),
            nb::arg("dim1"),
            nb::arg("dim2"),
            nb::arg("pad_value") = 0.0f));
}
}  // namespace ttnn::operations::data_movement::detail
