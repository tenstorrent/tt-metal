// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "width_to_height_shard_device_operation_types.hpp"
#include "width_to_height_shard_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct WidthToHeightShardDeviceOperation {
    using operation_attributes_t = WidthToHeightShardParams;
    using tensor_args_t = WidthToHeightShardInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<WidthToHeightShardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor width_to_height_shard(
    const Tensor& input_tensor,
    const tt::tt_metal::CoreRangeSet& output_core_range_set,
    const std::optional<Tensor>& preallocated_output);

}  // namespace ttnn::prim
