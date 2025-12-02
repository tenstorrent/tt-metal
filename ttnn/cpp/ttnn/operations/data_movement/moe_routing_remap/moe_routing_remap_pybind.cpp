// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
// #include <tt-metalium/sub_device_types.hpp>
// #include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "moe_routing_remap.hpp"
#include "moe_routing_remap_pybind.hpp"

namespace ttnn::operations::data_movement::detail {

void py_bind_moe_routing_remap(py::module& module) {
    const auto* doc = R"doc(

Remap MoE routing weights to local device routing weights. Partitions groups of non-zero weights, which may be
non-uniformly distributed, and maps them to devices along the specified cluster axis.

for example:,

non_zero_weight_size=8,
expert_parallel_size=4,
Total experts= 32
routing_weights_tensor (1,total_experts): [0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 4, 5, 0, 0, 0, 6,0, 0, 0, 7, 0, 8, 0, 0, 0, 0, 0, 0, 0]

Each device will have 2 non-zero values in their output.

cluster device 0: [0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cluster device 1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cluster device 2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 6,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cluster device 3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 7, 0, 8, 0, 0, 0, 0, 0, 0, 0]

Args:
    routing_weights_tensor (ttnn.Tensor): tensor of weights for selected experts, replicated on all devices `[1, total_experts]`
    non_zero_weight_size (integer): Total number of selected experts, non-zero weights in routing_weights_tensor.
    expert_parallel_size (integer): Number of devices in a cluster.
    cluster_axis (integer): Device mesh axis of cluster, 0: columns, 1: rows.

Keyword Args:
    memory_config (ttnn.MemoryConfig, optional): Optional memory configuration for the output. Defaults to `None`.
    optional_output_tensor (ttnn.Tensor, optional): Optional output buffer.

Returns:
    ttnn.Tensor: Tensor containing the device partitioned local weights, `[devices/devices, total_experts]`

    )doc";

    using OperationType = decltype(ttnn::moe_routing_remap);
    ttnn::bind_registered_operation(
        module,
        ttnn::moe_routing_remap,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& routing_weights_tensor,
               const uint32_t non_zero_weight_size,
               const uint32_t expert_parallel_size,
               const uint32_t cluster_axis,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& optional_output_tensor) {
                return self(
                    routing_weights_tensor,
                    non_zero_weight_size,
                    expert_parallel_size,
                    cluster_axis,
                    memory_config,
                    optional_output_tensor);
            },
            py::arg("routing_weights_tensor").noconvert(),
            py::arg("non_zero_weight_size"),
            py::arg("expert_parallel_size"),
            py::arg("cluster_axis"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("optional_output_tensor") = std::nullopt,
        });
}

}  // namespace ttnn::operations::data_movement::detail
