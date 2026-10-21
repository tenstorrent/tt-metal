// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "entry_correction_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

void EntryCorrectionOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view name = "entry_correction";
    for (const auto* state : {&in.final_a, &in.final_b, &in.entry, &in.tail_entry}) {
        kda_factory_detail::check_dtype(*state, tt::tt_metal::DataType::FLOAT32, name, "the states");
        TT_FATAL(state->logical_shape() == in.final_b.logical_shape(), "{}: the states must share one shape", name);
    }
    kda_factory_detail::check_actual_start(in.final_b, in.actual_start, name);
    const uint32_t heads = in.final_b.logical_shape()[0];
    const uint32_t chunks = attrs.local_rows / tt::constants::TILE_HEIGHT;
    TT_FATAL(
        !attrs.cores.empty() && attrs.cores.size() % heads == 0 && chunks % (attrs.cores.size() / heads) == 0,
        "{}: the cores must split evenly over the heads, and a head's cores over its {} chunks",
        name,
        chunks);
    for (const auto* terms : {&in.zero_output, &in.identity_output}) {
        const auto& shard = terms->memory_config().shard_spec();
        TT_FATAL(
            shard.has_value() && shard->grid.num_cores() == attrs.cores.size() &&
                shard->shape[0] == chunks / (attrs.cores.size() / heads) * tt::constants::TILE_HEIGHT,
            "{}: the chunk terms must be sharded so each core's chunk rows are in its L1",
            name);
    }
}

EntryCorrectionOperation::spec_return_value_t EntryCorrectionOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto spec =
        [](const ttnn::Shape& shape, tt::tt_metal::DataType dtype, const tt::tt_metal::MemoryConfig& memory) {
            return tt::tt_metal::TensorSpec(
                shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), memory));
        };
    return {
        spec(in.zero_output.logical_shape(), tt::tt_metal::DataType::BFLOAT16, attrs.output_mem_config),
        spec(in.final_b.logical_shape(), tt::tt_metal::DataType::FLOAT32, ttnn::DRAM_MEMORY_CONFIG)};
}

EntryCorrectionOperation::tensor_return_value_t EntryCorrectionOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.final_b.device()));
    }
    return outputs;
}

}  // namespace ttnn::experimental::prim
