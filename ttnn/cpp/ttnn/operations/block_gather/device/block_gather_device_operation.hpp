// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "block_gather_device_operation_types.hpp"
#include "block_gather_program_factory.hpp"

namespace ttnn::operations::block_gather {

// Call order on a launch: create_output_tensors, compute_program_hash, then on a hit
// validate_on_program_cache_hit and the factory's override_runtime_arguments, on a miss
// validate_on_program_cache_miss and create_descriptor.
struct BlockGatherDeviceOperation {
    using operation_attributes_t = BlockGatherParams;
    using tensor_args_t = BlockGatherInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<BlockGatherProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Every check depends only on the attributes and the tensor specs, which the key fixes: a hit checks nothing.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // The Python's key: block, the grid, the scores / ids specs and the output's memory config; whether the caller
    // gave the output is left out (one program either way).
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The Python entry's asserts, in its order; the public entry calls it before the output is allocated.
void validate_inputs(const BlockGatherParams& attrs, const BlockGatherInputs& t);

}  // namespace ttnn::operations::block_gather

namespace ttnn::prim {

Tensor block_gather(
    const Tensor& scores,
    const Tensor& ids,
    const std::optional<Tensor>& output,
    const ttnn::operations::block_gather::BlockGatherParams& params);

}  // namespace ttnn::prim
