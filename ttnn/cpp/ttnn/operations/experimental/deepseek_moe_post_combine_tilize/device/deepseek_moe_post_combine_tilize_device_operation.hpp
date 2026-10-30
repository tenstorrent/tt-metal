// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "deepseek_moe_post_combine_tilize_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEPostCombineTilizeDeviceOperation {
    using operation_attributes_t = DeepseekMoEPostCombineTilizeParams;
    using tensor_args_t = DeepseekMoEPostCombineTilizeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;

    // The only per-dispatch state is the input tensor (a tensor binding) and the sharded output
    // tensor backing the output DFB (a borrowed-memory DFB); the framework patches both on a
    // cache hit, so there is no override.
    struct DeepseekMoEPostCombineTilizeProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };
    using program_factory_t = std::variant<DeepseekMoEPostCombineTilizeProgramFactory>;

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor deepseek_moe_post_combine_tilize(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_memory_config);

}  // namespace ttnn::prim
