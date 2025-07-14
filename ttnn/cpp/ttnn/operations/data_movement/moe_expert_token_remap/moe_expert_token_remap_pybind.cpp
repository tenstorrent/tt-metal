// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

#include "moe_expert_token_remap.hpp"
#include "moe_expert_token_remap_pybind.hpp"

namespace ttnn::operations::data_movement::detail {

void py_bind_moe_expert_token_remap(py::module& module) {
    auto doc = R"doc()doc";

    using OperationType = decltype(ttnn::moe_expert_token_remap);
    ttnn::bind_registered_operation(
        module,
        ttnn::moe_expert_token_remap,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& topk_tensor,
               const ttnn::Tensor& expert_mapping_tensor,
               const ttnn::Tensor& expert_metadata_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& optional_output_tensor,
               QueueId queue_id) {
                return self(
                    queue_id,
                    topk_tensor,
                    expert_mapping_tensor,
                    expert_metadata_tensor,
                    memory_config,
                    optional_output_tensor);
            },
            py::arg("topk_tensor").noconvert(),
            py::arg("expert_indices_tensor").noconvert(),
            py::arg("expert_mapping_tensor").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("optional_output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::data_movement::detail
