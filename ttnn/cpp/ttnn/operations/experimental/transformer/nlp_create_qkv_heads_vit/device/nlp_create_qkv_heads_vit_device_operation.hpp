// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateHeadsVitDeviceOperation {
    using operation_attributes_t = NlpCreateQkvHeadsVitParams;
    using tensor_args_t = NlpCreateQkvHeadsVitInputs;
    using spec_return_value_t = NlpCreateQkvHeadsVitResultSpec;
    using tensor_return_value_t = NlpCreateQkvHeadsVitResult;

    // The only per-dispatch state is the four buffer addresses (input, q, k, v); the work split and
    // every per-core tile offset derive from the input's padded shape and the compute grid, both
    // covered by the program hash. The address bindings are therefore the whole cache-hit refresh.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::vector<Tensor> nlp_create_qkv_heads_vit(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors = std::nullopt);
}  // namespace ttnn::prim
