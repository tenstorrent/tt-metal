// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_matmul_prefill_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr std::string_view op_name = "all_gather_matmul_prefill";

void check_tile_interleaved(const Tensor& tensor, std::string_view name) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.is_allocated(),
        "{}: {} must be an allocated device tensor",
        op_name,
        name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{}: {} must be TILE layout", op_name, name);
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "{}: {} must be interleaved",
        op_name,
        name);
    const auto& shape = tensor.padded_shape();
    TT_FATAL(shape.rank() >= 2, "{}: {} must be [..., rows, columns]", op_name, name);
    for (int i = 0; i + 2 < static_cast<int>(shape.rank()); ++i) {
        TT_FATAL(shape[i] == 1, "{}: {}'s leading dimensions must be 1", op_name, name);
    }
}

}  // namespace

AllGatherMatmulPrefillOperation::program_factory_t AllGatherMatmulPrefillOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AllGatherMatmulPrefillProgramFactory{};
}

void AllGatherMatmulPrefillOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& in) {
    TT_FATAL(
        in.input.is_allocated() && in.weight.is_allocated() && in.gathered.is_allocated(),
        "{}: the tensors must be allocated",
        op_name);
}

void AllGatherMatmulPrefillOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_tile_interleaved(in.input, "input");
    check_tile_interleaved(in.weight, "weight");
    check_tile_interleaved(in.gathered, "gathered");
    TT_FATAL(
        in.input.dtype() == in.gathered.dtype() && in.input.dtype() == DataType::BFLOAT16,
        "{}: input and gathered must be BFLOAT16",
        op_name);
    TT_FATAL(
        in.gathered.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "{}: gathered must be in DRAM",
        op_name);
    const auto* mesh = in.input.device();
    TT_FATAL(attrs.cluster_axis < mesh->shape().dims(), "{}: invalid cluster_axis", op_name);
    const uint32_t ranks = mesh->shape()[attrs.cluster_axis];
    TT_FATAL(ranks >= 2, "{}: the cluster axis must hold at least two devices", op_name);
    const auto& input = in.input.padded_shape();
    const auto& gathered = in.gathered.padded_shape();
    const auto& weight = in.weight.padded_shape();
    TT_FATAL(
        gathered[-2] == input[-2] && gathered[-1] == input[-1] * ranks,
        "{}: gathered must be [rows, ranks x the input's columns]",
        op_name);
    TT_FATAL(weight[-2] == gathered[-1], "{}: the weight's rows must be the gathered in0's columns", op_name);
    TT_FATAL(attrs.counter_addresses.size() == ranks - 1, "{}: pass one slice counter per other rank", op_name);
    TT_FATAL(attrs.num_links >= 1, "{}: num_links must be positive", op_name);

    const auto& config = attrs.program_config;
    const auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t row_tiles = input[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t slice_tiles = input[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(!config.transpose_mcast, "{}: transpose_mcast is not supported", op_name);
    TT_FATAL(
        config.compute_with_storage_grid_size.x < grid.x,
        "{}: the matmul grid must leave a worker column to the gather",
        op_name);
    TT_FATAL(
        config.per_core_M * config.compute_with_storage_grid_size.y == row_tiles &&
            config.out_block_h == config.per_core_M,
        "{}: every matmul row must hold per_core_M of the rows, in one block",
        op_name);
    TT_FATAL(config.out_block_w == config.per_core_N, "{}: one output block per core", op_name);
    TT_FATAL(
        slice_tiles % config.in0_block_w == 0,
        "{}: a rank's K slice ({} tiles) must be whole K blocks of {}",
        op_name,
        slice_tiles,
        config.in0_block_w);
    TT_FATAL(
        2 * attrs.num_links <= grid.y,
        "{}: the gather needs two senders per link in the free column of {} cores",
        op_name,
        grid.y);
}

AllGatherMatmulPrefillOperation::spec_return_value_t AllGatherMatmulPrefillOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    // The input's shape with the weight's columns.
    auto shape = in.input.logical_shape();
    shape[-1] = in.weight.logical_shape()[-1];
    return {tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            attrs.output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), attrs.output_mem_config))};
}

AllGatherMatmulPrefillOperation::tensor_return_value_t AllGatherMatmulPrefillOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(attrs, in).at(0), in.input.device())};
}

Tensor all_gather_matmul_prefill(
    const AllGatherMatmulPrefillParams& params, const AllGatherMatmulPrefillInputs& inputs) {
    return ttnn::device_operation::launch<AllGatherMatmulPrefillOperation>(params, inputs).at(0);
}

}  // namespace ttnn::experimental::prim
