// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moe_compute_nanobind.hpp"
#include "moe_compute.hpp"
#include "device/kernels/moe_ring_common.h"
#include "device/hostdevcommon/config.hpp"

#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_moe_compute(nb::module_& mod) {
    // Bind the activation function enum
    nb::enum_<ttnn::experimental::prim::detail::MoEActivationFunction>(mod, "MoEActivationFunction")
        .value("SILU", ttnn::experimental::prim::detail::MoEActivationFunction::SILU)
        .value("SWIGLU", ttnn::experimental::prim::detail::MoEActivationFunction::SWIGLU);
    ttnn::bind_function<"moe_compute", "ttnn.experimental.">(
        mod,
        R"doc(
        Experimental, high-performance MoE operation for DeepSeek.
        )doc",
        &ttnn::experimental::moe_compute,
        nb::arg("tilize_input_tensor").noconvert(),
        nb::arg("tilize_expert_indices_tensor").noconvert(),
        nb::arg("tilize_expert_scores_tensor").noconvert(),
        nb::arg("tilize_expert_mapping_tensor").noconvert(),
        nb::arg("matmul_w0_w1_tensor").noconvert(),
        nb::arg("matmul_w2_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("layer_id"),
        nb::arg("output_height_shard_dim"),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("topology") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("mux_core_range_set") = nb::none(),
        nb::arg("output_memory_config") = nb::none(),
        nb::arg("optional_output_tensor") = nb::none(),
        nb::arg("optional_cross_device_semaphore") = nb::none(),
        nb::arg("activation_type") = nb::none());
}

void bind_get_moe_combine_cores(nb::module_& mod) {
    const auto* doc = R"doc(Return the ordered list of cores assigned to A2A Combine for the MoE module flow )doc";
    ttnn::bind_function<"get_moe_combine_cores", "ttnn.experimental.">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<ttnn::MeshDevice*, const uint32_t, const uint32_t>(
                &ttnn::experimental::get_moe_combine_cores),
            nb::arg("mesh_device"),
            nb::arg("combine_token_parallel_cores"),
            nb::arg("combine_data_parallel_cores")));
}
}  // namespace ttnn::operations::experimental::ccl
