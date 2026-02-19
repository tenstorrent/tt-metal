// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "selective_reduce_combine_device_operation_types.hpp"
#include "selective_reduce_combine_program_factory.hpp"

namespace ttnn::experimental::prim {

struct SelectiveReduceCombineDeviceOperation {
    using operation_attributes_t = SelectiveReduceCombineParams;
    using tensor_args_t = SelectiveReduceCombineTensors;

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = ttnn::Tensor;

    using program_factory_t = std::variant<UnifiedSelectReduce>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    // static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Empty as there doesn't seem to be any complicated hashing requirement
    // static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
ttnn::Tensor selective_reduce_combine(
    const ttnn::Tensor& dense_input_tensor,
    const ttnn::Tensor& dense_metadata_tensor,
    const ttnn::Tensor& dense_token_maps_tensor,
    const ttnn::Tensor& dense_token_counts_tensor,
    const uint32_t hidden_size,
    const uint32_t batch_size,
    const uint32_t seq_size,
    const uint32_t select_experts_k,
    const uint32_t experts,
    const std::optional<uint32_t>& axis,
    tt::tt_fabric::Topology topology,
    const uint32_t num_links,
    const uint32_t num_token_parallel_cores,
    const uint32_t num_data_parallel_cores,
    const std::vector<ttnn::CoreCoord>& worker_core_range_set,
    const CoreRangeSet& mux_core_range_set,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<GlobalSemaphore>& optional_cross_device_semaphore);
}  // namespace ttnn::prim
