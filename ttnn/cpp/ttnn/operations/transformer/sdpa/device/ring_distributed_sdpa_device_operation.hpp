// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "ring_distributed_sdpa_device_operation_types.hpp"
#include "ring_distributed_sdpa_program_factory.hpp"

namespace ttnn::operations::transformer::ring_distributed_sdpa {

struct RingDistributedSdpaDeviceOperation {
    using operation_attributes_t = ring_distributed_sdpa::operation_attributes_t;
    using tensor_args_t = ring_distributed_sdpa::tensor_args_t;
    using spec_return_value_t = ring_distributed_sdpa::spec_return_value_t;
    using tensor_return_value_t = ring_distributed_sdpa::tensor_return_value_t;

    using program_factory_t = std::variant<program::RingDistributedSdpaMeshWorkloadFactory>;
    using shared_variables_t = program::RingDistributedSdpaMeshWorkloadFactory::shared_variables_t;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::transformer::ring_distributed_sdpa

namespace ttnn::prim {

ttnn::operations::transformer::ring_distributed_sdpa::RingDistributedSdpaDeviceOperation::tensor_return_value_t
ring_distributed_sdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    uint32_t ring_size,
    std::optional<uint32_t> ring_id,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<ttnn::Tensor>& page_table = std::nullopt,
    std::optional<int64_t> chunk_start_idx = std::nullopt);

}  // namespace ttnn::prim
