// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
// #include <tt-metalium/sub_device_types.hpp>
// #include <tt-metalium/fabric_edm_types.hpp>

#include "moe_routing_remap.hpp"
#include "moe_routing_remap_pybind.hpp"

namespace ttnn::operations::data_movement::detail {

void py_bind_moe_routing_remap(py::module& module) {
    auto doc = R"doc(

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
