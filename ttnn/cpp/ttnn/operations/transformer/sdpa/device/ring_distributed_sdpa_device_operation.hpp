// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"

#include "ring_distributed_sdpa_device_operation_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct RingDistributedSdpaDeviceOperation {
    using operation_attributes_t = RingDistributedSDPAParams;
    using tensor_args_t = RingDistributedSDPAInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct RingDistributedSdpaProgramFactory {
        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);
    };

    using program_factory_t = std::variant<RingDistributedSdpaProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor);
};

Tensor ring_distributed_sdpa(
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
