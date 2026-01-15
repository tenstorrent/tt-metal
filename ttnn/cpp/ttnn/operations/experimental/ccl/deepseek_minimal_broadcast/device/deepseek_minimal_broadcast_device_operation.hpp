// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/deepseek_minimal_broadcast_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/deepseek_minimal_broadcast_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <optional>
#include <variant>
#include <vector>

namespace ttnn::operations::experimental::ccl::deepseek_minimal_broadcast {

struct DeepseekMinimalBroadcastDeviceOperation {
    using operation_attributes_t = DeepseekMinimalBroadcastParams;
    using tensor_args_t = DeepseekMinimalBroadcastInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::DeepseekMinimalBroadcastProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::ccl::deepseek_minimal_broadcast

namespace ttnn::prim {

ttnn::operations::experimental::ccl::deepseek_minimal_broadcast::DeepseekMinimalBroadcastDeviceOperation::
    tensor_return_value_t
    deepseek_minimal_broadcast(
        const ttnn::Tensor& input_tensor,
        const MeshCoordinate& sender_coord,
        uint32_t num_links,
        const std::optional<MemoryConfig>& memory_config,
        tt::tt_fabric::Topology topology,
        std::optional<uint32_t> cluster_axis,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

}  // namespace ttnn::prim
