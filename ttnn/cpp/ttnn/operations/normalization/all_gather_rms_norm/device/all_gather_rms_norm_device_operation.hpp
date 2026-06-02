// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device.hpp>

#include "ttnn/operations/normalization/all_gather_rms_norm/device/all_gather_rms_norm_device_operation_types.hpp"
#include "ttnn/operations/normalization/all_gather_rms_norm/device/all_gather_rms_norm_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct AllGatherRMSNormDeviceOperation {
    using operation_attributes_t = AllGatherRMSNormParams;
    using tensor_args_t = AllGatherRMSNormInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<AllGatherRMSNormProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

ttnn::Tensor all_gather_rms_norm(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& global_semaphore,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    float epsilon,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    ttnn::ccl::Topology topology,
    std::optional<size_t> num_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::Tensor>& persistent_stats_tensor,
    uint32_t num_heads = 1);

}  // namespace ttnn::prim
