// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"

namespace ttnn::operations::reduction::detail {
void bind_reduction_argmax_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Argmax. Returns indices of maximum values.
            Output is UINT32, ROW_MAJOR, INTERLEAVED (DRAM or L1).

            Args:
                input_tensor (ttnn.Tensor): On-device, INTERLEAVED input.

            Keyword args:
                dim (int, optional): Dim to reduce. ``None`` reduces all elements (ROW_MAJOR input only). Default: ``None``.
                keepdim (bool, optional): Keep reduced dim. Default: ``False``.
                sub_core_grids (CoreRangeSet, optional): Limits execution to a subset of cores. Supported on ROW_MAJOR last-dim reductions (<= 2 ranges) and batch/channel dim reductions. Default: ``None``.
                memory_config (ttnn.MemoryConfig, optional): Output memory (INTERLEAVED DRAM/L1). Default: input's memory_config.
                output_tensor (ttnn.Tensor, optional): Preallocated output (must be UINT32, ROW_MAJOR, INTERLEAVED, same device). Default: ``None``.

            Supported:

            - **dim=None** (reduce all elements):
              - input layout: ROW_MAJOR
              - dtypes: BFLOAT16/FLOAT32/INT32/UINT32/UINT16

            - **dim = rank-1** (last / width):
              - ROW_MAJOR input: BFLOAT16/FLOAT32/INT32/UINT32/UINT16 (multi-core by default)
              - TILE input: BFLOAT16/FLOAT32 (single-core)

            - **dim = rank-2** (height):
              - BFLOAT16/FLOAT32 only
              - ROW_MAJOR inputs are internally tilized; this path runs single-core

            - **0 <= dim < rank-2** (batch/channel dims, rank >= 3):
              - BFLOAT16/FLOAT32 only (integer dtypes not supported)
              - input may be ROW_MAJOR or TILE (ROW_MAJOR is converted to TILE internally)
              - output is produced in TILE internally and converted to ROW_MAJOR
              - ``sub_core_grids`` is supported (pass a single-core ``CoreRangeSet`` to run on one core)

            Not supported:

            - Sharded tensors (inputs/outputs must be INTERLEAVED)
            - TILE input with ``dim=None``
            - Batch/channel dim reductions with INT/UINT inputs
            - Integer dtypes on batch/channel dim reductions
        )doc";

    ttnn::bind_function<"argmax">(
        mod,
        doc,
        &ttnn::argmax,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dim") = nb::none(),
        nb::arg("keepdim") = false,
        nb::kw_only(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::reduction::detail
