// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_combine_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "all_to_all_combine.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void bind_all_to_all_combine(nb::module_& mod) {
    auto doc = R"doc()doc";

    using OperationType = decltype(ttnn::all_to_all_combine);
    ttnn::bind_registered_operation(
        mod,
        ttnn::all_to_all_combine,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& expert_mapping_tensor,
               const ttnn::Tensor& expert_metadata_tensor,
               const GlobalSemaphore& global_semaphore,
               const bool local_reduce,
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
                    local_reduce,
                    num_links,
                    topology,
                    memory_config,
                    axis,
                    subdevice_id,
                    optional_output_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("expert_indices_tensor").noconvert(),
            nb::arg("expert_mapping_tensor").noconvert(),
            nb::arg("global_semaphore"),
            nb::kw_only(),
            nb::arg("local_reduce") = false,
            nb::arg("num_links") = 1,
            nb::arg("topology") = tt::tt_fabric::Topology::Linear,
            nb::arg("memory_config") = nb::none(),
            nb::arg("axis") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none(),
            nb::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::ccl
