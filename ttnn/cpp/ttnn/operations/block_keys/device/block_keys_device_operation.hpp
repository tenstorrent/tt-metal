// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "block_keys_device_operation_types.hpp"
#include "block_keys_program_factory.hpp"

namespace ttnn::operations::block_keys {

// Every attribute is structural; compute_program_hash adds the pos / ids specs and leaves out only whether the
// output is the caller's.
struct BlockKeysDeviceOperation {
    using operation_attributes_t = BlockKeysParams;
    using tensor_args_t = BlockKeysInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<BlockKeysProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // The key fixes every spec and attribute the miss validator checked, and the public entry checks
    // them before every launch: nothing is left for a hit.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The Python entry's asserts, in its order (TT_FATAL).
void validate_inputs(const BlockKeysParams& attrs, const BlockKeysInputs& t);

}  // namespace ttnn::operations::block_keys

namespace ttnn::prim {

Tensor block_keys(
    const Tensor& pos,
    const Tensor& ids,
    const std::optional<Tensor>& output,
    const ttnn::operations::block_keys::BlockKeysParams& params);

}  // namespace ttnn::prim
