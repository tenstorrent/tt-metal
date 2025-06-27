// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_combine_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "all_to_all_combine.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void py_bind_all_to_all_combine(py::module& module) {
    auto doc = R"doc()doc";

    using OperationType = decltype(ttnn::all_to_all_combine);
    ttnn::bind_registered_operation(
        module,
        ttnn::all_to_all_combine,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& expert_mapping_tensor,
               const ttnn::Tensor& expert_metadata_tensor,
               const GlobalSemaphore& global_semaphore,
               const uint32_t num_links,
               const tt::tt_fabric::Topology topology,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<uint32_t>& axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::Tensor>& optional_output_tensor,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    expert_mapping_tensor,
                    expert_metadata_tensor,
                    global_semaphore,
                    num_links,
                    topology,
                    memory_config,
                    axis,
                    subdevice_id,
                    optional_output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("expert_indices_tensor").noconvert(),
            py::arg("expert_mapping_tensor").noconvert(),
            py::arg("global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("topology") = tt::tt_fabric::Topology::Linear,
            py::arg("memory_config") = std::nullopt,
            py::arg("axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("optional_output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::ccl
