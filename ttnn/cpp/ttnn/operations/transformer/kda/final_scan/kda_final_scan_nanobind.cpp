// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_final_scan_nanobind.hpp"

#include "kda_final_scan.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::transformer {

void bind_kda_final_scan(nb::module_& mod) {
    ttnn::bind_function<"kda_final_chunk_scan", "ttnn.transformer.">(
        mod,
        R"doc(Apply the final recurrent scan to KDA chunk-preparation tensors.)doc",
        &ttnn::transformer::kda_final_chunk_scan,
        nb::arg("v_beta").noconvert(),
        nb::arg("kd").noconvert(),
        nb::arg("q_decay").noconvert(),
        nb::arg("intra").noconvert(),
        nb::arg("k_dec_t").noconvert(),
        nb::arg("final_decay").noconvert(),
        nb::arg("t_inv").noconvert(),
        nb::kw_only(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("chunk_size") = 32,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("state_only") = false,
        nb::arg("identity_tile") = nb::none(),
        nb::arg("summary_pair") = false,
        nb::arg("output_bf16") = false);
}

}  // namespace ttnn::operations::transformer
