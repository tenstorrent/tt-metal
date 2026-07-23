// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/kda_recurrent/device/kda_recurrent_device_operation_types.hpp"
#include "ttnn/operations/transformer/kda_recurrent/device/kda_recurrent_program_factory.hpp"

namespace ttnn::prim {

struct KDARecurrentDeviceOperation {
    using operation_attributes_t = KDARecurrentParams;
    using tensor_args_t = KDARecurrentInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<KDARecurrentProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> kda_recurrent_step(
    const Tensor& q_scaled,
    const Tensor& k_unit,
    const Tensor& v,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& state,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
