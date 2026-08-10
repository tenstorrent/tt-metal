// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "attn_res_merge_device_operation_types.hpp"
#include "attn_res_merge_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct AttnResMergeDeviceOperation {
    using operation_attributes_t = AttnResMergeParams;
    using tensor_args_t = AttnResMergeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<AttnResMergeProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor attn_res_merge(
    const Tensor& partial,
    const Tensor& prefix_sum,
    const Tensor& shift,
    const Tensor& mass,
    const Tensor& live_scores,
    uint32_t site,
    uint32_t num_partials,
    float inv_hidden_size,
    float eps,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
