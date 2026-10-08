// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "exchange_histories_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

ExchangeHistoriesOperation::program_factory_t ExchangeHistoriesOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ExchangeHistoriesProgramFactory{};
}

void ExchangeHistoriesOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "exchange_histories";
    TT_FATAL(
        attrs.local_rows > 0 && attrs.local_rows % tt::constants::TILE_HEIGHT == 0,
        "{}: local_rows must be positive and 32-aligned",
        operation_name);
    kda_factory_detail::check_actual_start(in.projected, in.actual_start, operation_name);
    if (in.actual_end) {
        kda_factory_detail::check_actual_start(in.actual_start, *in.actual_end, operation_name);
    }
    kda_factory_detail::check_allocated_device_tensor(in.projected, operation_name, "projected");
    kda_factory_detail::check_layout(in.projected, tt::tt_metal::Layout::TILE, operation_name, "projected");
    kda_factory_detail::check_dtype(in.projected, tt::tt_metal::DataType::BFLOAT16, operation_name, "projected");
    kda_factory_detail::check_interleaved(in.projected, operation_name, "projected");
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    const auto& shape = in.projected.padded_shape();
    TT_FATAL(
        shape.volume() / shape[-1] == attrs.local_rows,
        "{}: projected must hold local_rows rows, got shape {}",
        operation_name,
        in.projected.logical_shape());
    TT_FATAL(
        attrs.width > 0 && attrs.width % tt::constants::TILE_WIDTH == 0 && attrs.width <= shape[-1],
        "{}: width must be a positive multiple of {} within the projection's {} columns",
        operation_name,
        tt::constants::TILE_WIDTH,
        shape[-1]);
    const auto* mesh = in.projected.device();
    TT_FATAL(attrs.sequence_parallel_axis < mesh->shape().dims(), "{}: invalid sequence_parallel_axis", operation_name);
    TT_FATAL(
        mesh->shape()[attrs.sequence_parallel_axis] > 1, "{}: needs several sequence-parallel ranks", operation_name);
}

ExchangeHistoriesOperation::spec_return_value_t ExchangeHistoriesOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, kda_chronology::selection::history_rows, attrs.width}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            attrs.output_mem_config));
    return {spec, spec};
}

ExchangeHistoriesOperation::tensor_return_value_t ExchangeHistoriesOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.projected.device()));
    }
    return outputs;
}

std::vector<Tensor> exchange_histories(
    const Tensor& projected,
    const tt::tt_metal::MemoryConfig& memory_config,
    const Tensor& actual_start,
    const std::optional<Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t width,
    tt::tt_fabric::Topology topology) {
    return ttnn::device_operation::launch<ExchangeHistoriesOperation>(
        ExchangeHistoriesParams{
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .width = width,
            .topology = topology,
            .output_mem_config = memory_config},
        ExchangeHistoriesInputs{.projected = projected, .actual_start = actual_start, .actual_end = actual_end});
}

}  // namespace ttnn::experimental::prim
