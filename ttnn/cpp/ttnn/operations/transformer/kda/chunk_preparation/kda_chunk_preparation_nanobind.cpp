// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_chunk_preparation_nanobind.hpp"

#include "kda_chunk_preparation.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::transformer {

void bind_kda_chunk_preparation(nb::module_& mod) {
    ttnn::bind_function<"kda_chunk_preparation", "ttnn.transformer.">(
        mod,
        R"doc(Prepare KDA chunk-local tensors for the final recurrent scan.)doc",
        &ttnn::transformer::kda_chunk_preparation,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("eye").noconvert(),
        nb::arg("tril").noconvert(),
        nb::arg("ones").noconvert(),
        nb::arg("masks").noconvert(),
        nb::kw_only(),
        nb::arg("chunk_size") = 32,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("v_flat") = false,
        nb::arg("value_heads") = 0,
        nb::arg("normalize_qk") = false,
        nb::arg("scale") = 1.0F,
        nb::arg("qk_flat") = false,
        nb::arg("key_heads") = 0,
        nb::arg("gate_flat") = false,
        nb::arg("output_bf16_mask") = 0);
}

}  // namespace ttnn::operations::transformer
