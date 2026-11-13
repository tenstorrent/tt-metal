// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"

#include "moe_tail_device_operation_types.hpp"
#include "moe_tail_program_factory.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

struct MoETailDeviceOperation {
    using operation_attributes_t = MoETailParams;
    using tensor_args_t = MoETailInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<TailEngineFactory, TailAllToAllFactory>;

    // kind, plan_key, fidelity, ring_slots, output memory config and the input specs: never an address.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::moe_block::moe_tail

namespace ttnn::prim {

Tensor moe_block_tail(
    const ttnn::operations::experimental::moe_block::moe_tail::MoETailParams& params,
    ttnn::operations::experimental::moe_block::moe_tail::MoETailInputs inputs);

}  // namespace ttnn::prim
