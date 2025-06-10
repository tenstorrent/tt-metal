// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather_pybind.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.hpp"

namespace ttnn::operations::experimental::ccl {

namespace py = pybind11;

void bind_fused_rms_1_1_32_8192(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::fused_rms_1_1_32_8192,
        R"doc(Only works for sharded shape (1,1,32,8192) sharded on 1 core
        )doc",
        // Stats is internally computed
        ttnn::pybind_arguments_t{
            // Used by all
            py::arg("input_tensor"),
            py::arg("program_config"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("global_semaphore"),  // TODO: Build this internally
            py::kw_only(),
            // all gather
            py::arg("persistent_output_tensor") = std::nullopt,
            py::arg("num_links") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Linear,
            py::arg("subdevice_id") = std::nullopt,
            // common
            py::arg("dtype") = std::nullopt,  // Should default to BFLOAT 16 on pre, nullopt on post
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            // on pre only
            py::arg("residual_input_tensor") = std::nullopt,
            // on post only
            py::arg("epsilon") = 1e-12,  // constant 1e-12 on pre, value only affects post
            py::arg("weight") = std::nullopt,
            py::arg("stats") = std::nullopt});
}
}  // namespace ttnn::operations::experimental::ccl
