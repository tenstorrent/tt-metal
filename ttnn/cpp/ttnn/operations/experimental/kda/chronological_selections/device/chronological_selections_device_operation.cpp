// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chronological_selections_device_operation.hpp"
#include "kernels/chronology.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
namespace ttnn::experimental::prim {
ChronologicalSelectionsOperation::program_factory_t ChronologicalSelectionsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChronologicalSelectionsFactory{};
}
void ChronologicalSelectionsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& in) {
    kda_factory_detail::check_actual_start(in.actual_start, in.actual_start, "chronological_selections");
    if (in.actual_end) {
        kda_factory_detail::check_actual_start(in.actual_start, *in.actual_end, "chronological_selections");
    }
    TT_FATAL(
        a.sequence_parallel_axis < in.actual_start.device()->shape().dims() && a.local_rows > 0 &&
            a.local_rows % tt::constants::TILE_HEIGHT == 0,
        "chronological_selections: invalid partition geometry");
    TT_FATAL(a.batch_heads > 0 && a.key_dim > 0 && a.value_dim > 0, "chronological_selections: invalid state geometry");
}
ChronologicalSelectionsOperation::spec_return_value_t ChronologicalSelectionsOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t&) {
    return {tt::tt_metal::TensorSpec(
        Shape({kda_chronology::selection::record_count, kda_chronology::selection::record_width}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            ttnn::DRAM_MEMORY_CONFIG))};
}
ChronologicalSelectionsOperation::tensor_return_value_t ChronologicalSelectionsOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(a, in)[0], in.actual_start.device())};
}

SelectRequestHistoryOperation::program_factory_t SelectRequestHistoryOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SelectRequestHistoryFactory{};
}
void SelectRequestHistoryOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& in) {
    constexpr auto name = "select_request_history";
    for (const auto* tensor : {&in.projected_qkv, &in.layer_history, &in.predecessor_history, &in.selection_records}) {
        kda_factory_detail::check_allocated_device_tensor(*tensor, name, "input");
        kda_factory_detail::check_same_device(in.projected_qkv, *tensor, name, "input");
        kda_factory_detail::check_layout(*tensor, tt::tt_metal::Layout::ROW_MAJOR, name, "input");
        kda_factory_detail::check_interleaved(*tensor, name, "input");
    }
    for (const auto* tensor : {&in.projected_qkv, &in.layer_history, &in.predecessor_history}) {
        kda_factory_detail::check_dtype(*tensor, tt::tt_metal::DataType::BFLOAT16, name, "history");
    }
    kda_factory_detail::check_dtype(in.selection_records, tt::tt_metal::DataType::UINT32, name, "selection_records");
    kda_factory_detail::check_actual_start(in.projected_qkv, in.actual_start, name);
    const auto& shape = in.projected_qkv.logical_shape();
    TT_FATAL(
        shape.rank() == 3 && shape[0] == 1 && shape[1] >= 3 && shape[2] > 0 && shape[2] % 32 == 0,
        "select_request_history requires [1, rows>=3, aligned width] QKV");
    TT_FATAL(
        in.layer_history.logical_shape() == Shape({1, kda_chronology::selection::history_rows, shape[2]}) &&
            in.predecessor_history.logical_shape() == in.layer_history.logical_shape(),
        "select_request_history requires matching [1, 3, width] histories");
    TT_FATAL(
        in.selection_records.logical_shape() ==
            Shape({kda_chronology::selection::record_count, kda_chronology::selection::record_width}),
        "select_request_history requires a chronological selection table");
}
SelectRequestHistoryOperation::spec_return_value_t SelectRequestHistoryOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& in) {
    return {tt::tt_metal::TensorSpec(
        in.layer_history.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            ttnn::DRAM_MEMORY_CONFIG))};
}
SelectRequestHistoryOperation::tensor_return_value_t SelectRequestHistoryOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(a, in)[0], in.projected_qkv.device())};
}

}  // namespace ttnn::experimental::prim
