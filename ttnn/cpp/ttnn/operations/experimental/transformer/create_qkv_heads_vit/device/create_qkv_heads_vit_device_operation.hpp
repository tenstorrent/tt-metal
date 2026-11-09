// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operations/experimental/transformer/create_qkv_heads_vit/device/create_qkv_heads_vit_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct CreateHeadsVitDeviceOperation {
    using operation_attributes_t = CreateQkvHeadsVitParams;
    using tensor_args_t = CreateQkvHeadsVitInputs;
    using spec_return_value_t = CreateQkvHeadsVitResultSpec;
    using tensor_return_value_t = CreateQkvHeadsVitResult;

    struct CreateQkvHeadsVitProgramFactory {
        // The only per-dispatch state is the four buffer addresses (input, q, k, v); the work split and
        // every per-core tile offset derive from the input's padded shape and the compute grid, both
        // covered by the program hash. The tensor bindings are therefore the whole cache-hit refresh.
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };
    using program_factory_t = std::variant<CreateQkvHeadsVitProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::vector<Tensor> create_qkv_heads_vit(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors = std::nullopt);
}  // namespace ttnn::prim
