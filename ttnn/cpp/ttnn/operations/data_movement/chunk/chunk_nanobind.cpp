// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_nanobind.hpp"

#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/chunk/chunk.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_chunk(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Splits `input_tensor` into chunks along dimension `dim`.

        Every chunk but the last has ``ceil(input_tensor.shape[dim] / chunks)`` elements along `dim`, the chunk size,
        and the last chunk holds the rest. So fewer than `chunks` chunks can come back: a dimension of 5 split with
        ``chunks=4`` gives chunks of 2, 2 and 1.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device. A TILE input must use 32x32 tiles.
            chunks (int): the number of chunks to split into. Must be at least 1.
            dim (int): the dimension to split along, in ``[-r, r)`` for an input of rank ``r``. Negative values count from the last dimension.

        Returns:
            List[ttnn.Tensor]: the chunks, in order along `dim`, with the dtype, layout and memory configuration of `input_tensor`.

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

            ``BFLOAT8_B`` tensors need a rank of 2 or more and, when `dim` is one of the last two dimensions, a chunk
            size that is a multiple of 32.

            ``BFLOAT16`` TILE input values must not be NaN when the input has rank 1, or when `dim` is one of the last
            two dimensions and the chunk size is not a multiple of 32.

        Memory Support:
            - Interleaved: DRAM and L1
        )doc";

    ttnn::bind_function<"chunk">(mod, doc, &ttnn::chunk, nb::arg("input_tensor"), nb::arg("chunks"), nb::arg("dim"));
}

}  // namespace ttnn::operations::data_movement
