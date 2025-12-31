// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "rms_allgather.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::ccl {

void bind_fused_rms_1_1_32_8192(nb::module_& mod) {
    mod.def(
        "fused_rms_1_1_32_8192",
        &ttnn::fused_rms_1_1_32_8192,
        R"doc(Only works for sharded shape (1,1,32,8192) sharded on 1 core
        )doc",
        nb::arg("input_tensor"),
        nb::arg("program_config"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("global_semaphore"),
        nb::kw_only(),
        nb::arg("persistent_output_tensor") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("topology") = ttnn::ccl::Topology::Linear,
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("residual_input_tensor") = nb::none(),
        nb::arg("epsilon") = 1e-12,
        nb::arg("weight") = nb::none(),
        nb::arg("stats") = nb::none(),
        nb::arg("use_noc1_only") = false);
}

}  // namespace ttnn::operations::experimental::ccl
