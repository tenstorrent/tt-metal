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
                input_tensor (ttnn.Tensor): the input tensor, on device. A TILE input must use 32x32 tiles.
                split_size (Union[int, List[int]]): the size of a single chunk, or a list of per-chunk sizes.
                dim (int): the dimension to split along, in ``[-r, r)`` for an input of rank ``r``. Negative values count from the last dimension. Defaults to 0.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensors. A sharded configuration without a shard spec gets one derived from the per-chunk shape, or falls back to DRAM interleaved when none can be derived. Defaults to `None`, which uses the input tensor's memory configuration, with these exceptions: for a sharded input, the shard spec is rescaled to the chunk shape, and the outputs are DRAM interleaved when the chunks are unequal, the shard spec cannot be rescaled, or the rescaled TILE shard is not a multiple of 32 in both dims; and an L1 input split into more than 2 chunks gives DRAM interleaved outputs when the chunks and the slice buffers are estimated not to fit in L1. A split into 1 or 2 chunks has no such fallback, so its L1 outputs can fail to allocate.

            Returns:
                List[ttnn.Tensor]: the chunks, in order along :attr:`dim`. Each chunk has the dtype and layout of :attr:`input_tensor` and the input's shape except along :attr:`dim`. The chunks are copies, except that on the :func:`ttnn.slice` path a single chunk covering the whole input, with an unchanged memory configuration, is the input tensor itself. A single chunk produced by the TILE kernel (see the note below) is still a copy.

            Note:
                Supported dtypes and layouts:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16
                     - TILE, ROW_MAJOR
                   * - UINT8
                     - ROW_MAJOR
                   * - BFLOAT8_B
                     - TILE

                ``BFLOAT16`` TILE input values must not be NaN when :attr:`dim` is one of the last two dims and a chunk
                starts at an index that is not a multiple of 32.

                Each chunk is produced by :func:`ttnn.slice`, except for an equal split of a TILE tensor along its last
                dim, which uses a single kernel when all of these hold, with the input viewed as rank 4 ``[N, C, H, W]``:

                * ``W`` is a multiple of 32, ``H`` and ``W`` are both at least 64, and the number of tiles in ``W``
                  is divisible by the number of chunks.
                * The number of chunks is at most the number of rows in the device's compute grid.
                * For a split into 2 chunks, ``N * C`` is less than or equal to the number of columns in the
                  compute grid. For any other number of chunks, ``N`` is 1 and ``C`` is less than or equal to the
                  number of columns.

                Otherwise the split falls back to :func:`ttnn.slice`.

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input and output, in DRAM or L1
                - ND sharded: input, with DRAM interleaved outputs by default. A split into 2 chunks that uses the TILE kernel with ``N`` greater than 1 is not supported for an ND sharded input; convert the input to interleaved first
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
