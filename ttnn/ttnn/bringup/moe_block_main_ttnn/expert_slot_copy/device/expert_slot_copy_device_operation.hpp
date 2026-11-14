// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "expert_slot_copy_device_operation_types.hpp"
#include "expert_slot_copy_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy {

struct SlotCopyDeviceOperation {
    using operation_attributes_t = SlotCopyParams;
    using tensor_args_t = SlotCopyInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<SlotCopyProgramFactory>;

    // Structure only: plan_key, the hashed attributes and the tensor specs. No address, no job.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy

namespace ttnn::prim {

// dst_gu is returned (written in place).
Tensor moe_block_main_expert_slot_copy(
    const ttnn::operations::bringup::moe_block_main::expert_slot_copy::SlotCopyInputs& inputs,
    const ttnn::operations::bringup::moe_block_main::expert_slot_copy::SlotCopyParams& params);

}  // namespace ttnn::prim
