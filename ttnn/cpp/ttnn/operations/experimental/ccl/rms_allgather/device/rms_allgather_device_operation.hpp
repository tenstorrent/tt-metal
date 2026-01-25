// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rms_allgather_device_operation_types.hpp"
#include "rms_allgather_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::experimental::prim {

namespace layernorm = ttnn::prim;

struct RMSAllGatherDeviceOperation {
    using operation_attributes_t = RMSAllGatherParams;
    using tensor_args_t = RMSAllGatherInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RMSAllGatherMeshWorkloadFactory>;
    using shared_variables_t = RMSAllGatherMeshWorkloadFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::RMSAllGatherDeviceOperation::tensor_return_value_t rms_allgather(
    const Tensor& input_tensor,
    const ttnn::prim::LayerNormProgramConfig& program_config,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    std::optional<size_t> num_preferred_links,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<const DataType> dtype,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& stats,
    bool use_noc1_only);

}  // namespace ttnn::prim
