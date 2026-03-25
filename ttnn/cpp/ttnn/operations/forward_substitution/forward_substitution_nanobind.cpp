// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "forward_substitution_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "forward_substitution.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::forward_substitution {

void bind_forward_substitution_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(

        Computes the forward substitution (I + A)^{-1} for a batched strict lower triangular
        matrix A. This is the fused on-device equivalent of the row-by-row sequential solve
        used in the chunked delta rule attention mechanism.

        The input matrix A has shape [batch, C, C] where C is the chunk size. A is expected
        to be lower triangular (values on and below the diagonal). The operation performs
        the sequential row-by-row forward substitution and adds the identity matrix.

        Args:
            input (ttnn.Tensor): Input tensor of shape [batch, C, C] in ROW_MAJOR layout, FLOAT32 dtype.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for output. Defaults to input's config.

        Returns:
            ttnn.Tensor: Output tensor of shape [batch, C, C] containing (I + A)^{-1}.

        Note:
            Supported configurations:

            .. list-table::
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR

            Memory Support:
                - Interleaved: DRAM and L1

            Limitations:
                - Input must be on device, ROW_MAJOR, FLOAT32, INTERLEAVED
                - Last two dimensions must be equal (square matrix)
                - Matrix size C must be a multiple of 8 and fit in L1 (C*C*4 + 3*C*4 <= 256KB)
    )doc";

    ttnn::bind_function<"forward_substitution">(
        mod, doc, &ttnn::forward_substitution, nb::arg("input"), nb::kw_only(), nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::forward_substitution
