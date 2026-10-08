// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_final_carry_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

SelectFinalCarryOperation::program_factory_t SelectFinalCarryOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SelectFinalCarryProgramFactory{};
}

void SelectFinalCarryOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "select_final_carry";
    TT_FATAL(
        attrs.local_rows > 0 && attrs.local_rows % tt::constants::TILE_HEIGHT == 0,
        "{}: local_rows must be positive and 32-aligned",
        operation_name);
    kda_factory_detail::check_actual_start(in.rank_final, in.actual_start, operation_name);
    if (in.actual_end) {
        kda_factory_detail::check_actual_start(in.actual_start, *in.actual_end, operation_name);
    }
    for (const auto* tensor : {&in.rank_final, &in.prefix_final}) {
        kda_factory_detail::check_allocated_device_tensor(*tensor, operation_name, "input");
        kda_factory_detail::check_layout(*tensor, tt::tt_metal::Layout::TILE, operation_name, "input");
        kda_factory_detail::check_dtype(*tensor, tt::tt_metal::DataType::FLOAT32, operation_name, "input");
        kda_factory_detail::check_interleaved(*tensor, operation_name, "input");
    }
    kda_factory_detail::check_same_device(in.rank_final, in.prefix_final, operation_name, "prefix_final");
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    const auto& tail = in.rank_final.logical_shape();
    const auto& prefix = in.prefix_final.logical_shape();
    TT_FATAL(prefix.rank() == 3, "{}: prefix_final must be [B*H, K, V]", operation_name);
    TT_FATAL(
        tail == prefix || (tail.rank() == 4 && tail[0] == prefix[0] && tail[2] == prefix[1] && tail[3] == prefix[2]),
        "{}: rank_final must be [B*H, K, V], or [B*H, groups, K, V] whose last group is the final state",
        operation_name);
    const auto* mesh = in.rank_final.device();
    TT_FATAL(attrs.sequence_parallel_axis < mesh->shape().dims(), "{}: invalid sequence_parallel_axis", operation_name);
    TT_FATAL(attrs.num_links > 0, "{}: num_links must be positive", operation_name);
}

SelectFinalCarryOperation::spec_return_value_t SelectFinalCarryOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {tt::tt_metal::TensorSpec(
        in.prefix_final.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            attrs.output_mem_config))};
}

SelectFinalCarryOperation::tensor_return_value_t SelectFinalCarryOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(attrs, in)[0], in.prefix_final.device())};
}

Tensor select_final_carry(
    const Tensor& rank_final,
    const Tensor& prefix_final,
    const tt::tt_metal::MemoryConfig& memory_config,
    const Tensor& actual_start,
    const std::optional<Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t num_links,
    tt::tt_fabric::Topology topology) {
    return ttnn::device_operation::launch<SelectFinalCarryOperation>(
        SelectFinalCarryParams{
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .num_links = num_links,
            .topology = topology,
            .output_mem_config = memory_config},
        SelectFinalCarryInputs{
            .rank_final = rank_final,
            .prefix_final = prefix_final,
            .actual_start = actual_start,
            .actual_end = actual_end})[0];
}

}  // namespace ttnn::experimental::prim
