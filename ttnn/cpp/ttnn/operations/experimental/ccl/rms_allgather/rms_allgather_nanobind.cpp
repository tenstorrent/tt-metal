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

void bind_fused_rms_minimal(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::fused_rms_minimal,
        R"doc(
            This fuses pre RMS, all gather, post rms, residual add, gamma operations, and output resharding into one.
            Requires a pre-allocated persistent tensor for the intermediate all gather that is tiled,
                shape per device (32,32) and sharded with a shard shape (32,32) on core (0,0)
            Requires that input the tensor be of shape (1,1,32,M) where M is a multiple of 32
        )doc",
        // Stats is internally computed
        ttnn::nanobind_arguments_t{// Used by all
                                   nb::arg("input_tensor"),
                                   nb::arg("program_config"),
                                   nb::arg("cluster_axis"),
                                   nb::arg("mesh_device"),
                                   nb::arg("global_semaphore"),
                                   nb::kw_only(),
                                   // all gather
                                   nb::arg("persistent_output_tensor") = nb::none(),
                                   nb::arg("num_links") = nb::none(),
                                   nb::arg("topology") = ttnn::ccl::Topology::Linear,
                                   nb::arg("subdevice_id") = nb::none(),
                                   // common
                                   nb::arg("dtype") = nb::none(),
                                   nb::arg("compute_kernel_config") = nb::none(),
                                   nb::arg("memory_config") = nb::none(),
                                   // on pre only
                                   nb::arg("residual_input_tensor") = nb::none(),
                                   // on post only
                                   nb::arg("epsilon") = 1e-12,
                                   nb::arg("weight") = nb::none(),
                                   nb::arg("stats") = nb::none(),
                                   nb::arg("use_noc1_only") = false});
}
}  // namespace ttnn::operations::experimental::ccl
