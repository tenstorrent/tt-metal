// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "block_max_bf16_device_operation_types.hpp"
#include "block_max_bf16_program_factory.hpp"

namespace ttnn::operations::block_max_bf16 {

// Call order (ttnn/api/ttnn/device_operation.hpp): create_output_tensors, compute_program_hash, then on a hit
// validate_on_program_cache_hit + the factory's override_runtime_arguments, on a miss
// validate_on_program_cache_miss + select_program_factory + create_descriptor.
struct BlockMaxBf16DeviceOperation {
    using operation_attributes_t = BlockMaxBf16Params;
    using tensor_args_t = BlockMaxBf16Inputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<BlockMaxReduceProgramFactory, BlockMaxStripProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Every check depends only on what the key holds (attributes but n, the tensor specs): nothing left to check.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The Python op's assertions, in its order, as TT_FATAL. has_n: whether the caller gave n.
void validate_inputs(const BlockMaxBf16Params& attrs, const BlockMaxBf16Inputs& t, bool has_n);

}  // namespace ttnn::operations::block_max_bf16

namespace ttnn::prim {

Tensor block_max_bf16(
    const Tensor& input,
    const std::optional<Tensor>& n_tensor,
    const std::optional<Tensor>& output,
    const ttnn::operations::block_max_bf16::BlockMaxBf16Params& params);

}  // namespace ttnn::prim
