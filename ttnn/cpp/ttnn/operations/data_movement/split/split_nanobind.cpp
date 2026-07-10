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

            This matches the semantics of :func:`torch.split`:

            * If :attr:`split_size` is an ``int``, the tensor is split into equally sized chunks of
              ``split_size`` along :attr:`dim`. If the dimension is not evenly divisible by
              ``split_size``, the last chunk is smaller. The number of outputs is
              ``ceil(input_tensor.shape[dim] / split_size)``.
            * If :attr:`split_size` is a ``list[int]``, the tensor is split into ``len(split_size)``
              chunks whose sizes along :attr:`dim` are given by the list entries. The sizes must sum
              exactly to ``input_tensor.shape[dim]``; an error is raised otherwise.

            .. note::
                The following degenerate cases differ from :func:`torch.split` (they involve
                zero-volume tensors, which are not supported by the device kernels):

                * A zero-size split dimension (``input_tensor.shape[dim] == 0``) with an integer
                  :attr:`split_size` raises an error instead of returning a single empty chunk.
                * Zero-size entries in a :attr:`split_size` list (e.g. ``[0, 10]``) raise an error;
                  every chunk size must be greater than 0.

            Equivalent pytorch code:

            .. code-block:: python

                # split_size as an int
                output_tensors = ttnn.split(input_tensor, 2, dim=1)
                # equivalent to
                output_tensors = torch.split(input_tensor, 2, dim=1)

                # split_size as a list
                output_tensors = ttnn.split(input_tensor, [2, 3], dim=1)
                # equivalent to
                output_tensors = torch.split(input_tensor, [2, 3], dim=1)

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
