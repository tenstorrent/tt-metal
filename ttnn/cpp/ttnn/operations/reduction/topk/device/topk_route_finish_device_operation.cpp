// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_route_finish_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include <limits>

namespace ttnn::operations::reduction::topk_route_finish {

namespace {

void validate_static_args(const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;

    // These fields are part of the program hash; they are validated on cache miss before compiling
    // the program and do not need to be rechecked on every cache hit.
    TT_FATAL(input.layout() == Layout::TILE, "topk_route_finish logits input must be TILE layout");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "topk_route_finish logits input must be BFLOAT16");
    TT_FATAL(!input.memory_config().is_sharded(), "topk_route_finish logits input must use interleaved memory");
    TT_FATAL(indices.layout() == Layout::ROW_MAJOR, "topk_route_finish indices input must be ROW_MAJOR");
    TT_FATAL(indices.dtype() == DataType::UINT32, "topk_route_finish indices input must be UINT32");
    TT_FATAL(!indices.memory_config().is_sharded(), "topk_route_finish indices input must use interleaved memory");
}

void validate_runtime_args(const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "topk_route_finish logits input must be on device");
    TT_FATAL(input.buffer() != nullptr, "topk_route_finish logits input must have an allocated buffer");
    TT_FATAL(indices.storage_type() == StorageType::DEVICE, "topk_route_finish indices input must be on device");
    TT_FATAL(indices.buffer() != nullptr, "topk_route_finish indices input must have an allocated buffer");

    const auto& input_shape = input.logical_shape();
    const auto& indices_shape = indices.logical_shape();
    TT_FATAL(input_shape.rank() >= 2, "topk_route_finish logits input must have rank >= 2");
    TT_FATAL(
        indices_shape.rank() == input_shape.rank(),
        "topk_route_finish inputs must have equal rank, got {} and {}",
        input_shape.rank(),
        indices_shape.rank());
    // The indices tensor is the logits' shape with the last dim swapped for k_rounded: the reader
    // pages index sticks with (batch * R + row), so every leading dim (R included) must match.
    for (int i = 0; i < static_cast<int>(input_shape.rank()) - 1; ++i) {
        TT_FATAL(
            indices_shape[i] == input_shape[i],
            "topk_route_finish shape mismatch at dim {}: logits {}, indices {}",
            i,
            input_shape[i],
            indices_shape[i]);
    }
    const uint32_t k_rounded = indices_shape[-1];
    TT_FATAL(
        k_rounded >= 16 && k_rounded % 16 == 0,
        "topk_route_finish k_rounded must be a positive multiple of 16, got {}",
        k_rounded);
    TT_FATAL(
        k_rounded <= input_shape[-1],
        "topk_route_finish k_rounded {} must be <= the logits width {}",
        k_rounded,
        input_shape[-1]);
}

}  // namespace

void TopkRouteFinishDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_runtime_args(attrs, tensor_args);
}

void TopkRouteFinishDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_static_args(attrs, tensor_args);
    validate_runtime_args(attrs, tensor_args);
}

TopkRouteFinishDeviceOperation::program_factory_t TopkRouteFinishDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return program::TopkRouteFinishProgramFactory{};
}

ttsl::hash::hash_t TopkRouteFinishDeviceOperation::compute_program_hash(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;
    const auto grid = input.device()->compute_with_storage_grid_size();

    // Program-structure terms (see the program factory's work split):
    //   - width_tiles (W_p / 32) feeds the reader's source-page math (compile arg);
    //   - k_rounded fixes K_t = div_up(k_rounded, 32) (compile arg) AND the indices input
    //     TensorAccessor's compile-time stick (page) size (k_rounded * 4 B);
    //   - total_tile_rows (padded volume / W_p / 32) fixes the unit count and hence the
    //     split_blocks_for_tilize core partition (which cores carry kernels is create-time state);
    //   - index_is_u32 selects the output index dtype, its CB/staging sizes, and the output
    //     TensorAccessor's page size (2048 vs 4096 B tiles).
    // Logical R stays hash-free: it only feeds reader/writer runtime args (the per-unit valid-row
    // clamp), re-derived from the tensors in override_runtime_arguments on every cache hit.
    const auto& padded = input.padded_shape();
    const uint32_t width_tiles = padded[-1] / tt::constants::TILE_WIDTH;
    const uint32_t total_tile_rows = (input.physical_volume() / padded[-1]) / tt::constants::TILE_HEIGHT;
    const uint32_t k_rounded = indices.logical_shape()[-1];
    const bool index_is_u32 = padded[-1] > std::numeric_limits<uint16_t>::max();

    return tt::tt_metal::operation::hash_operation<TopkRouteFinishDeviceOperation>(
        input.dtype(),
        input.layout(),
        input.memory_config().memory_layout(),
        input.memory_config().buffer_type(),
        indices.dtype(),
        indices.layout(),
        indices.memory_config().memory_layout(),
        indices.memory_config().buffer_type(),
        grid.x,
        grid.y,
        width_tiles,
        total_tile_rows,
        k_rounded,
        index_is_u32);
}

spec_return_value_t TopkRouteFinishDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;

    // Both outputs: the logits' logical shape with the last dim swapped for k_rounded, TILE,
    // same interleaved memory config as the logits — exactly what the unfused route's
    // to_layout(TILE) pair produced.
    ttnn::Shape output_shape = input.logical_shape();
    output_shape[-1] = indices.logical_shape()[-1];

    // Stock device-op index dtype contract (topk_device_operation.cpp compute_output_specs):
    // UINT16 iff the TILE-PADDED source width fits 16 bits, else UINT32.
    const DataType index_dtype =
        input.padded_shape()[-1] <= std::numeric_limits<uint16_t>::max() ? DataType::UINT16 : DataType::UINT32;

    return {
        tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), input.memory_config())),
        tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(index_dtype, tt::tt_metal::PageConfig(Layout::TILE), input.memory_config()))};
}

tensor_return_value_t TopkRouteFinishDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    return {create_device_tensor(std::get<0>(specs), device), create_device_tensor(std::get<1>(specs), device)};
}

std::tuple<TopkRouteFinishDeviceOperation::operation_attributes_t, TopkRouteFinishDeviceOperation::tensor_args_t>
TopkRouteFinishDeviceOperation::invoke(const Tensor& input_tensor, const Tensor& indices_tensor) {
    return {operation_attributes_t{}, tensor_args_t{.input_tensor = input_tensor, .indices_tensor = indices_tensor}};
}

}  // namespace ttnn::operations::reduction::topk_route_finish

namespace ttnn::operations::reduction::topk {

std::vector<Tensor> topk_route_finish(const Tensor& input_tensor, const Tensor& indices_tensor) {
    using Op = ttnn::operations::reduction::topk_route_finish::TopkRouteFinishDeviceOperation;
    auto [operation_attributes, tensor_args] = Op::invoke(input_tensor, indices_tensor);
    auto [values, indices] = ttnn::device_operation::launch<Op>(operation_attributes, tensor_args);
    std::vector<Tensor> result;
    result.reserve(2);
    result.push_back(std::move(values));
    result.push_back(std::move(indices));
    return result;
}

}  // namespace ttnn::operations::reduction::topk
