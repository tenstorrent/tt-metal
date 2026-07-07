// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_preprocess_device_operation_types.hpp"
#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_preprocess_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GatedDeltaAttnPreprocessDeviceOperation {
    using operation_attributes_t = GatedDeltaAttnPreprocessParams;
    using tensor_args_t = GatedDeltaAttnPreprocessInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<GatedDeltaAttnPreprocessProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> gated_delta_attn_preprocess(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& g,
    const Tensor& triu_ones,
    const Tensor& tril_mask,
    const Tensor& eye,
    const Tensor& lower_causal,
    const Tensor& eye_32,
    uint32_t chunk_size,
    float diag_alpha,
    bool bf16_value_path,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
