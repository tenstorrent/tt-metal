// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
            Splits :attr:`input_tensor` into chunks along dimension :attr:`dim` and returns them as a list of tensors.

            The behavior depends on the type of :attr:`split_size`:

            * If :attr:`split_size` is an ``int``, the tensor is split into contiguous chunks of
              ``split_size`` elements along :attr:`dim`. When ``input_tensor.shape[dim]`` is not an
              exact multiple of ``split_size``, the final chunk holds the remainder and is smaller.
              The number of outputs is ``ceil(input_tensor.shape[dim] / split_size)``.
            * If :attr:`split_size` is a ``list[int]``, the tensor is split into ``len(split_size)``
              contiguous chunks whose sizes along :attr:`dim` are the list entries, in order. The
              entries must sum exactly to ``input_tensor.shape[dim]``.

            Constraints:

            * Every chunk size must be greater than 0 (both a zero entry in a :attr:`split_size`
              list and a zero-size split dimension raise an error; zero-volume tensors are not
              supported by the device kernels).
            * For a :attr:`split_size` list, the entries must sum exactly to ``input_tensor.shape[dim]``;
              an error is raised for both under- and over-covering lists.

            Example:

            .. code-block:: python

                # int split_size: 6 along dim=1 -> chunks of size 2
                a, b, c = ttnn.split(input_tensor, 2, dim=1)  # input_tensor.shape[1] == 6

                # list split_size: explicit per-chunk sizes summing to shape[1]
                a, b = ttnn.split(input_tensor, [2, 4], dim=1)  # input_tensor.shape[1] == 6

            Args:
                * :attr:`input_tensor`: Input Tensor.
                * :attr:`split_size` (Union[int, list[int]]): Size of a single chunk, or a list of per-chunk sizes.
                * :attr:`dim` (int): Dimension along which to split. Negative indexing is supported. Defaults to 0.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensors.

            Returns:
                List[ttnn.Tensor]: The list of output tensors.
        )doc";

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

        // Overload 2: list of split_sizes (ttsl::SmallVector<int64_t>)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttsl::SmallVector<int64_t>&,
                int64_t,
                const std::optional<ttnn::MemoryConfig>&>(&ttnn::split),
            nb::arg("input_tensor"),
            nb::arg("split_size"),
            nb::arg("dim") = 0,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}
}  // namespace ttnn::operations::data_movement::detail
