// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_tile_rows_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

uint32_t select_tile_rows_count(const SelectTileRowsParams& attrs, const SelectTileRowsInputs& in);

SelectTileRowsOperation::program_factory_t SelectTileRowsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SelectTileRowsProgramFactory{};
}

void SelectTileRowsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "select_tile_rows";
    kda_factory_detail::check_allocated_device_tensor(in.input, operation_name, "input");
    kda_factory_detail::check_dtype(in.input, tt::tt_metal::DataType::BFLOAT16, operation_name, "input");
    kda_factory_detail::check_interleaved(in.input, operation_name, "input");
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    TT_FATAL(
        attrs.record.has_value() != in.indices.has_value(),
        "{}: pass exactly one of indices or a chronological selection record",
        operation_name);
    if (in.indices.has_value()) {
        kda_factory_detail::check_allocated_device_tensor(*in.indices, operation_name, "indices");
        kda_factory_detail::check_layout(*in.indices, tt::tt_metal::Layout::ROW_MAJOR, operation_name, "indices");
        kda_factory_detail::check_dtype(*in.indices, tt::tt_metal::DataType::UINT32, operation_name, "indices");
        kda_factory_detail::check_same_device(in.input, *in.indices, operation_name, "indices");
        TT_FATAL(
            in.indices->logical_shape().rank() == 1 && in.indices->buffer()->num_pages() == 1,
            "{}: indices must be one row-major UINT32 vector",
            operation_name);
        TT_FATAL(
            in.indices->logical_shape()[0] * sizeof(uint32_t) <= 64,
            "{}: at most {} rows are selected at once",
            operation_name,
            64 / sizeof(uint32_t));
    } else {
        using namespace kda_chronology::selection;
        TT_FATAL(
            *attrs.record < record_count && *attrs.record != final_state && *attrs.record != final_state + 1,
            "{}: record {} is not a history selection",
            operation_name,
            *attrs.record);
        TT_FATAL(in.actual_start.has_value(), "{}: a chronological selection needs actual_start", operation_name);
        kda_factory_detail::check_actual_start(in.input, *in.actual_start, operation_name);
        if (in.actual_end.has_value()) {
            kda_factory_detail::check_actual_start(in.input, *in.actual_end, operation_name);
        }
        TT_FATAL(
            attrs.local_rows > 0 && attrs.local_rows % tt::constants::TILE_HEIGHT == 0,
            "{}: local_rows must be positive and tile aligned",
            operation_name);
    }
    const uint32_t rows = select_tile_rows_count(attrs, in);
    TT_FATAL(
        attrs.rows_per_output == 0 || attrs.rows_per_output == rows || (attrs.rows_per_output * 2 == rows),
        "{}: rows_per_output must keep one output or split the rows into two",
        operation_name);
    const auto& shape = in.input.logical_shape();
    TT_FATAL(shape.rank() == 3 && shape[0] == 1, "{}: input must be [1, rows, columns]", operation_name);
    TT_FATAL(
        attrs.width > 0 && attrs.width % tt::constants::TILE_WIDTH == 0 && attrs.width <= shape[-1],
        "{}: width must be positive, tile aligned and within the input columns",
        operation_name);
    TT_FATAL(
        in.input.layout() == tt::tt_metal::Layout::TILE || attrs.width == shape[-1],
        "{}: a row-major input selects whole rows",
        operation_name);
}

SelectTileRowsOperation::spec_return_value_t SelectTileRowsOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const uint32_t rows = select_tile_rows_count(attrs, in);
    const uint32_t rows_per_output = attrs.rows_per_output == 0 ? rows : attrs.rows_per_output;
    spec_return_value_t specs;
    for (uint32_t first = 0; first < rows; first += rows_per_output) {
        specs.push_back(tt::tt_metal::TensorSpec(
            ttnn::Shape({1, rows_per_output, attrs.width}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                attrs.output_mem_config)));
    }
    return specs;
}

SelectTileRowsOperation::tensor_return_value_t SelectTileRowsOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.input.device()));
    }
    return outputs;
}

std::vector<Tensor> select_tile_rows(
    const Tensor& input,
    const std::optional<Tensor>& indices,
    uint32_t width,
    const tt::tt_metal::MemoryConfig& memory_config,
    std::optional<uint32_t> record,
    const std::optional<Tensor>& actual_start,
    const std::optional<Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t rows_per_output) {
    return ttnn::device_operation::launch<SelectTileRowsOperation>(
        SelectTileRowsParams{
            .width = width,
            .output_mem_config = memory_config,
            .record = record,
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .rows_per_output = rows_per_output},
        SelectTileRowsInputs{
            .input = input, .indices = indices, .actual_start = actual_start, .actual_end = actual_end});
}

}  // namespace ttnn::experimental::prim
