// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_single_user_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

constexpr uint32_t kCollapseCores = 8;
constexpr uint32_t kTotalCores = 10;

void validate_width_shard(const Tensor& tensor, uint32_t num_cores, uint32_t expected_width, const char* name) {
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED, "{} must be WIDTH_SHARDED", name);
    const auto& shard_spec = tensor.shard_spec();
    TT_FATAL(shard_spec.has_value(), "{} must have a shard specification", name);
    TT_FATAL(shard_spec->grid.num_cores() == num_cores, "{} must use {} cores", name, num_cores);
    const auto bbox = shard_spec->grid.bounding_box();
    const CoreCoord expected_start(0, 0);
    const CoreCoord expected_end(num_cores - 1, 0);
    TT_FATAL(
        bbox.start_coord == expected_start && bbox.end_coord == expected_end,
        "{} must use row-major cores [0,{}]",
        name,
        num_cores - 1);
    TT_FATAL(
        shard_spec->shape[1] == expected_width / num_cores,
        "{} shard width must be {}, got {}",
        name,
        expected_width / num_cores,
        shard_spec->shape[1]);
}

void validate_single_user_tensors(const FusedSingleUserParams& attributes, const FusedSingleUserInputs& tensor_args) {
    const auto& fused_w = tensor_args.fused_w;
    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& comb_bias = tensor_args.comb_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;

    for (const auto* tensor : {&fused_w, &pre_bias, &post_bias, &comb_bias, &hidden_streams}) {
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "all fused hyperconnection inputs must be on device");
        TT_FATAL(tensor->layout() == Layout::TILE, "all fused hyperconnection inputs must use TILE layout");
        TT_FATAL(tensor->dtype() == DataType::BFLOAT16, "all fused hyperconnection inputs must be BFLOAT16");
    }

    const uint32_t hc = attributes.num_streams;
    const auto& fused_shape = fused_w.logical_shape();
    const auto& hidden_shape = hidden_streams.logical_shape();
    const uint32_t packed_width = (2 + hc) * hc;
    TT_FATAL(hc >= 1 && hc <= 32, "num_streams must be in [1,32], got {}", hc);
    TT_FATAL(attributes.sinkhorn_iters >= 1, "sinkhorn_iters must be >= 1");
    TT_FATAL(
        packed_width <= tt::constants::TILE_WIDTH,
        "single-user fused_w currently packs into one tile; packed width {} exceeds {}",
        packed_width,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        fused_shape.rank() == 4 && fused_shape[0] == 1 && fused_shape[1] == 1 && fused_shape[2] == 1 &&
            fused_shape[3] >= packed_width,
        "fused_w must be [1,1,1,>=((2+H)*H)] with packed width {}, got {}",
        packed_width,
        fused_shape);
    TT_FATAL(
        fused_w.padded_shape()[-1] % 32 == 0,
        "fused_w padded width must be tile-aligned, got {}",
        fused_w.padded_shape()[-1]);
    TT_FATAL(hidden_shape.rank() == 4, "hidden_streams must be rank-4, got {}", hidden_shape.rank());
    TT_FATAL(
        hidden_shape[0] == 1 && hidden_shape[1] == 1 && hidden_shape[2] == hc,
        "hidden_streams must be [1,1,H,D] with H={}, got {}",
        hc,
        hidden_shape);
    const uint32_t d = static_cast<uint32_t>(hidden_shape[3]);
    TT_FATAL(d % (kCollapseCores * 32) == 0, "hidden size D must be divisible by {}", kCollapseCores * 32);
    TT_FATAL(
        pre_bias.logical_shape() == ttnn::Shape({1, 1, 1, hc}),
        "pre_bias must be [1,1,1,H], got {}",
        pre_bias.logical_shape());
    TT_FATAL(
        post_bias.logical_shape() == ttnn::Shape({1, 1, 1, hc}),
        "post_bias must be [1,1,1,H], got {}",
        post_bias.logical_shape());
    TT_FATAL(
        comb_bias.logical_shape() == ttnn::Shape({1, 1, hc, hc}),
        "comb_bias must be [1,1,H,H], got {}",
        comb_bias.logical_shape());

    validate_width_shard(fused_w, 1, static_cast<uint32_t>(fused_w.padded_shape()[-1]), "fused_w");
    validate_width_shard(hidden_streams, kCollapseCores, d, "hidden_streams");

    const auto device_grid = fused_w.device()->compute_with_storage_grid_size();
    TT_FATAL(
        device_grid.x >= kTotalCores, "single-user hyperconnection requires {} cores in the first row", kTotalCores);
}

}  // namespace

void FusedSingleUserDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_single_user_tensors(attributes, tensor_args);
}

void FusedSingleUserDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_single_user_tensors(attributes, tensor_args);
}

FusedSingleUserDeviceOperation::spec_return_value_t FusedSingleUserDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto make_layout = [&](const MemoryConfig& memory_config) {
        return tt::tt_metal::TensorLayout(
            tensor_args.fused_w.dtype(), tt::tt_metal::PageConfig(tensor_args.fused_w.layout()), memory_config);
    };
    const auto& hidden_shape = tensor_args.hidden_streams.logical_shape();
    const uint32_t d = static_cast<uint32_t>(hidden_shape[3]);
    return {
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, attributes.num_streams, 1}), make_layout(attributes.post_comb_output_mem_config)),
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, attributes.num_streams, attributes.num_streams}),
            make_layout(attributes.post_comb_output_mem_config)),
        tt::tt_metal::TensorSpec(ttnn::Shape({1, 1, 1, d}), make_layout(attributes.collapsed_output_mem_config))};
}

FusedSingleUserDeviceOperation::tensor_return_value_t FusedSingleUserDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(attributes, tensor_args);
    auto* device = tensor_args.fused_w.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device)};
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection

namespace ttnn::prim {

std::array<Tensor, 3> fused_hyperconnection_single_user(
    const Tensor& fused_w,
    const Tensor& pre_bias,
    const Tensor& post_bias,
    const Tensor& comb_bias,
    const Tensor& hidden_streams,
    uint32_t num_streams,
    uint32_t sinkhorn_iters,
    float pre_scale,
    float post_scale,
    float comb_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::deepseek::hyperconnection::FusedSingleUserDeviceOperation;
    const MemoryConfig post_comb_output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const MemoryConfig collapsed_output_mem_config = memory_config.value_or(hidden_streams.memory_config());
    auto operation_attributes = OperationType::operation_attributes_t{
        .num_streams = num_streams,
        .sinkhorn_iters = sinkhorn_iters,
        .pre_scale = pre_scale,
        .post_scale = post_scale,
        .comb_scale = comb_scale,
        .eps = eps,
        .post_comb_output_mem_config = post_comb_output_mem_config,
        .collapsed_output_mem_config = collapsed_output_mem_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .fused_w = fused_w,
        .pre_bias = pre_bias,
        .post_bias = post_bias,
        .comb_bias = comb_bias,
        .hidden_streams = hidden_streams,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
