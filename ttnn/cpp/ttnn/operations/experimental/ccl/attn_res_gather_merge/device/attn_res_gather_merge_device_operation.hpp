// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "attn_res_gather_merge_device_operation_types.hpp"
#include "attn_res_gather_merge_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct AttnResGatherMergeDeviceOperation {
    using operation_attributes_t = AttnResGatherMergeParams;
    using tensor_args_t = AttnResGatherMergeInputs;
    // The mixed hidden state, and — only where a `pending` write was handed in — the
    // settled stream behind it. The second output is absent rather than empty when
    // there is nothing to settle, so a caller that never defers writes allocates
    // nothing extra.
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<AttnResGatherMergeMeshWorkloadFactory>;
    using shared_variables_t = AttnResGatherMergeMeshWorkloadFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> attn_res_gather_merge(
    const Tensor& partial,
    const Tensor& prefix_sum,
    const Tensor& shift,
    const Tensor& mass,
    const Tensor& q,
    const Tensor& stats,
    const std::optional<Tensor>& pending,
    uint32_t site,
    float inv_hidden_size,
    float eps,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& semaphore,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
