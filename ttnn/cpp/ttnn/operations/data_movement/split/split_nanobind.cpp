// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>  // split returns a vector

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "split.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_split(nb::module_& mod) {
    const auto* doc =
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

    // Bind the free functions directly - no struct!
    ttnn::bind_function<"split">(
        mod,
        doc,

        // Overload 1: single split_size (int64_t)
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int64_t, int64_t, const std::optional<ttnn::MemoryConfig>&>(
                &ttnn::split),
            nb::arg("input_tensor"),
            nb::arg("split_size"),
            nb::arg("dim") = 0,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),

        // Overload 2: list of split_sizes (SmallVector<int64_t>)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::SmallVector<int64_t>&,
                int64_t,
                const std::optional<ttnn::MemoryConfig>&>(&ttnn::split),
            nb::arg("input_tensor"),
            nb::arg("split_size"),
            nb::arg("dim") = 0,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}
}  // namespace ttnn::operations::data_movement::detail
