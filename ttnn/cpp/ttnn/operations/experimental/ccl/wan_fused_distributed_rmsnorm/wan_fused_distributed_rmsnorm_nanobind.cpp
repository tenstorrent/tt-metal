// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "wan_fused_distributed_rmsnorm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/wan_fused_distributed_rmsnorm.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_wan_fused_distributed_rmsnorm(nb::module_& mod) {
    ttnn::bind_function<"wan_fused_distributed_rmsnorm", "ttnn.experimental.">(
        mod,
        R"doc(
            Fused distributed RMSNorm for Wan2.2 attention.

            Composes three stages into one call:
              1. `wan_fused_rmsnorm_pre_allgather` — per-row partial sum-of-squares in fp32.
              2. All-gather of the partial statistics across `cluster_axis`.
              3. `wan_fused_rmsnorm_post_allgather` — finalize normalization, optionally
                 split heads, apply RoPE, and cast output dtype.

            First-draft implementation chains the three existing primitives; intended to
            be replaced by a single device op later.
        )doc",
        &ttnn::experimental::wan_fused_distributed_rmsnorm,
        nb::arg("input_tensor"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("multi_device_global_semaphore"),
        nb::kw_only(),
        nb::arg("topology") = ttnn::ccl::Topology::Ring,
        nb::arg("epsilon") = 1e-5,
        nb::arg("num_heads_per_device") = 1,
        nb::arg("weight") = nb::none(),
        nb::arg("transformation_mat") = nb::none(),
        nb::arg("rope_cos") = nb::none(),
        nb::arg("rope_sin") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("persistent_output_buffer") = nb::none(),
        nb::arg("num_preferred_links") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("use_device_op") = false);
}

}  // namespace ttnn::operations::experimental::ccl
