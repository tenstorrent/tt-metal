// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_chunk_scan_nanobind.hpp"

#include "recurrent_chunk_scan.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::recurrent_chunk_scan::detail {

void bind_recurrent_chunk_scan(nb::module_& mod) {
    ttnn::bind_function<"recurrent_chunk_scan", "ttnn.experimental.kda.">(
        mod,
        R"doc(Apply the KDA recurrence over independently prepared chunks.)doc",
        &ttnn::experimental::kda::recurrent_chunk_scan,
        nb::arg("v_beta").noconvert(),
        nb::arg("kd").noconvert(),
        nb::arg("q_decay").noconvert(),
        nb::arg("intra").noconvert(),
        nb::arg("k_dec_t").noconvert(),
        nb::arg("final_decay").noconvert(),
        nb::arg("t_inv").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"summarize_chunk_recurrence", "ttnn.experimental.kda.">(
        mod,
        R"doc(Summarize prepared KDA chunks as one semantic affine transform.)doc",
        &ttnn::experimental::kda::summarize_chunk_recurrence,
        nb::arg("v_beta").noconvert(),
        nb::arg("kd").noconvert(),
        nb::arg("q_decay").noconvert(),
        nb::arg("intra").noconvert(),
        nb::arg("k_dec_t").noconvert(),
        nb::arg("final_decay").noconvert(),
        nb::arg("t_inv").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::recurrent_chunk_scan::detail
