// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_tile_rows_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

SelectTileRowsOperation::program_factory_t SelectTileRowsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SelectTileRowsProgramFactory{};
}

void SelectTileRowsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "select_tile_rows";
    kda_factory_detail::check_allocated_device_tensor(in.input, operation_name, "input");
    kda_factory_detail::check_layout(in.input, tt::tt_metal::Layout::TILE, operation_name, "input");
    kda_factory_detail::check_dtype(in.input, tt::tt_metal::DataType::BFLOAT16, operation_name, "input");
    kda_factory_detail::check_interleaved(in.input, operation_name, "input");
    kda_factory_detail::check_allocated_device_tensor(in.indices, operation_name, "indices");
    kda_factory_detail::check_layout(in.indices, tt::tt_metal::Layout::ROW_MAJOR, operation_name, "indices");
    kda_factory_detail::check_dtype(in.indices, tt::tt_metal::DataType::UINT32, operation_name, "indices");
    kda_factory_detail::check_same_device(in.input, in.indices, operation_name, "indices");
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    TT_FATAL(
        in.indices.logical_shape().rank() == 1 && in.indices.buffer()->num_pages() == 1,
        "{}: indices must be one row-major UINT32 vector",
        operation_name);
    TT_FATAL(
        in.indices.logical_shape()[0] * sizeof(uint32_t) <= 64,
        "{}: at most {} rows are selected at once",
        operation_name,
        64 / sizeof(uint32_t));
    const auto& shape = in.input.logical_shape();
    TT_FATAL(shape.rank() == 3 && shape[0] == 1, "{}: input must be [1, rows, columns]", operation_name);
    TT_FATAL(
        attrs.width > 0 && attrs.width % tt::constants::TILE_WIDTH == 0 && attrs.width <= shape[-1],
        "{}: width must be positive, tile aligned and within the input columns",
        operation_name);
}

SelectTileRowsOperation::spec_return_value_t SelectTileRowsOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {tt::tt_metal::TensorSpec(
        ttnn::Shape({1, in.indices.logical_shape()[0], attrs.width}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            attrs.output_mem_config))};
}

SelectTileRowsOperation::tensor_return_value_t SelectTileRowsOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(attrs, in)[0], in.input.device())};
}

Tensor select_tile_rows(
    const Tensor& input, const Tensor& indices, uint32_t width, const tt::tt_metal::MemoryConfig& memory_config) {
    return ttnn::device_operation::launch<SelectTileRowsOperation>(
        SelectTileRowsParams{.width = width, .output_mem_config = memory_config},
        SelectTileRowsInputs{.input = input, .indices = indices})[0];
}

}  // namespace ttnn::experimental::prim
