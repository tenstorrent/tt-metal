// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "attn_res_block_mix_device_operation_types.hpp"
#include "attn_res_block_mix_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct AttnResBlockMixDeviceOperation {
    using operation_attributes_t = AttnResBlockMixParams;
    using tensor_args_t = AttnResBlockMixInputs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 3>;
    using tensor_return_value_t = std::array<Tensor, 3>;
    using program_factory_t = std::variant<AttnResBlockMixProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 3> attn_res_block_mix(
    const std::vector<Tensor>& sealed,
    const Tensor& stats,
    uint32_t num_sites,
    float inv_hidden_size,
    float eps,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
