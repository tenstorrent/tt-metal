// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_route_prep_device_operation.hpp"

#include <tt-metalium/constants.hpp>

namespace ttnn::operations::reduction::topk_route_prep {

namespace {

void validate_static_args(const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    // These fields are part of the program hash; they are validated on cache miss before compiling
    // the program and do not need to be rechecked on every cache hit.
    TT_FATAL(input.layout() == Layout::TILE, "topk_route_prep input must be TILE layout");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "topk_route_prep input must be BFLOAT16");
    TT_FATAL(!input.memory_config().is_sharded(), "topk_route_prep input must use interleaved memory");
}

void validate_runtime_args(const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "topk_route_prep input must be on device");
    TT_FATAL(input.buffer() != nullptr, "topk_route_prep input must have an allocated buffer");
    TT_FATAL(input.logical_shape().rank() >= 2, "topk_route_prep input must have rank >= 2");
}

}  // namespace

void TopkRoutePrepDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_runtime_args(attrs, tensor_args);
}

void TopkRoutePrepDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_static_args(attrs, tensor_args);
    validate_runtime_args(attrs, tensor_args);
}

TopkRoutePrepDeviceOperation::program_factory_t TopkRoutePrepDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return program::TopkRoutePrepProgramFactory{};
}

ttsl::hash::hash_t TopkRoutePrepDeviceOperation::compute_program_hash(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto grid = input.device()->compute_with_storage_grid_size();

    // Program-structure terms, all W_p/R_p-derived (see the program factory's work split):
    //   - width_tiles (W_p / 32) fixes the block widths (bw_full/bw_last -> kernel compile args,
    //     CB page sizes) and the per-row block count;
    //   - total_tile_rows (padded volume / W_p / 32) fixes the total block count and hence the
    //     split_blocks_for_tilize core partition (which cores carry kernels is create-time state);
    //   - the logical width fixes the output TensorAccessor's compile-time stick (page) size.
    // Logical R stays hash-free: it only feeds writer runtime args (the partial-tile-height row
    // clamp), re-derived from the tensors in override_runtime_arguments on every cache hit.
    const auto& padded = input.padded_shape();
    const uint32_t width_tiles = padded[-1] / tt::constants::TILE_WIDTH;
    const uint32_t total_tile_rows = (input.physical_volume() / padded[-1]) / tt::constants::TILE_HEIGHT;

    return tt::tt_metal::operation::hash_operation<TopkRoutePrepDeviceOperation>(
        input.dtype(),
        input.layout(),
        input.memory_config().memory_layout(),
        input.memory_config().buffer_type(),
        grid.x,
        grid.y,
        width_tiles,
        total_tile_rows,
        input.logical_shape()[-1]);
}

spec_return_value_t TopkRoutePrepDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    // Same logical shape, ROW_MAJOR (logical sticks, tile padding dropped), same interleaved
    // memory config — exactly what to_layout(ROW_MAJOR) produced in the unfused route.
    return tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), input.memory_config()));
}

tensor_return_value_t TopkRoutePrepDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<TopkRoutePrepDeviceOperation::operation_attributes_t, TopkRoutePrepDeviceOperation::tensor_args_t>
TopkRoutePrepDeviceOperation::invoke(const Tensor& input_tensor) {
    return {operation_attributes_t{}, tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::reduction::topk_route_prep

namespace ttnn::operations::reduction::topk {

Tensor topk_route_prep(const Tensor& input_tensor) {
    using Op = ttnn::operations::reduction::topk_route_prep::TopkRoutePrepDeviceOperation;
    auto [operation_attributes, tensor_args] = Op::invoke(input_tensor);
    return ttnn::device_operation::launch<Op>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::reduction::topk
