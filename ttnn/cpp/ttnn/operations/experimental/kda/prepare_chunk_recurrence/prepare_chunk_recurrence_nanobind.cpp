// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prepare_chunk_recurrence_nanobind.hpp"

#include "prepare_chunk_recurrence.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::prepare_chunk_recurrence::detail {

void bind_prepare_chunk_recurrence(nb::module_& mod) {
    ttnn::bind_function<"prepare_chunk_recurrence", "ttnn.experimental.kda.">(
        mod,
        R"doc(Prepare KDA chunk-local tensors for the final recurrent scan.)doc",
        &ttnn::experimental::kda::prepare_chunk_recurrence,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("eye").noconvert(),
        nb::arg("tril").noconvert(),
        nb::arg("ones").noconvert(),
        nb::arg("masks").noconvert(),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_bf16_mask") = 0);
}

}  // namespace ttnn::operations::experimental::kda::prepare_chunk_recurrence::detail
