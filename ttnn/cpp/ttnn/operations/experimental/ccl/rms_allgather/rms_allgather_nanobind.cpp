// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_fused_rms_1_1_32_8192(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::fused_rms_1_1_32_8192,
        R"doc(Only works for sharded shape (1,1,32,8192) sharded on 1 core
        )doc",
        // Stats is internally computed
        ttnn::nanobind_arguments_t{
            // Used by all
            nb::arg("input_tensor"),
            nb::arg("program_config"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("global_semaphore"),  // TODO: Build this internally
            nb::kw_only(),
            // all gather
            nb::arg("persistent_output_tensor") = nb::none(),
            nb::arg("num_links") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Linear,
            nb::arg("subdevice_id") = nb::none(),
            // common
            nb::arg("dtype") = nb::none(),  // Should default to BFLOAT 16 on pre, nullopt on post
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            // on pre only
            nb::arg("residual_input_tensor") = nb::none(),
            // on post only
            nb::arg("epsilon") = 1e-12,  // constant 1e-12 on pre, value only affects post
            nb::arg("weight") = nb::none(),
            nb::arg("stats") = nb::none(),
            nb::arg("use_noc1_only") = false});
}
}  // namespace ttnn::operations::experimental::ccl
