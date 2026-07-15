// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include "masked_per_token_cast_back_device_operation_types.hpp"
#include "masked_per_token_cast_back_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::masked_per_token_cast_back {

struct MaskedPerTokenCastBackDeviceOperation {
    using operation_attributes_t = MaskedPerTokenCastBackParams;
    using tensor_args_t = MaskedPerTokenCastBackInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MaskedPerTokenCastBackProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim::masked_per_token_cast_back

namespace ttnn::prim {
ttnn::Tensor masked_per_token_cast_back(
    const Tensor& input_e4m3,
    const Tensor& input_scale,
    const Tensor& expert_region_offsets,
    const Tensor& expert_token_counts,
    const Tensor& global_expert_idx_table,
    uint32_t experts_per_chip,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    bool scales_from_metadata = false);
}  // namespace ttnn::prim
