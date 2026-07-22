// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/operations/transformer/flash_kda/device/flash_kda_device_operation_types.hpp"
#include "ttnn/operations/transformer/flash_kda/device/flash_kda_program_factory.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Device operation returning two tensors: [S_new, out].
struct FlashKdaDeviceOperation {
    using operation_attributes_t = FlashKdaParams;
    using tensor_args_t = FlashKdaInputs;

    // Return two Tensors: S_new [N, Dk, Dv] and out [N, 1, Dv].
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<FlashKdaProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Low-level dispatch function (used by public API).
std::vector<Tensor> flash_kda(
    const Tensor& S_prev,
    const Tensor& g,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& q,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
