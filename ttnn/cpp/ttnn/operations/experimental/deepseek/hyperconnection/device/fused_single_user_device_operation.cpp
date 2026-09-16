// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_single_user_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

void validate_fused_w_shard(const Tensor& fused_w) {
    TT_FATAL(
        fused_w.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED, "fused_w must be WIDTH_SHARDED");
    const auto& shard_spec = fused_w.shard_spec();
    TT_FATAL(shard_spec.has_value(), "fused_w must have a shard specification");
    TT_FATAL(shard_spec->grid.num_cores() == 1, "fused_w must use 1 core");
    TT_FATAL(
        shard_spec->grid.contains(CoreCoord{0, 0}),
        "fused_w must live on logical core (0,0) so collapse can broadcast it");
    const uint32_t expected_width = static_cast<uint32_t>(fused_w.padded_shape()[-1]);
    TT_FATAL(
        shard_spec->shape[1] == expected_width,
        "fused_w shard width must be {}, got {}",
        expected_width,
        shard_spec->shape[1]);
}

void validate_hidden_shard(const Tensor& hidden, uint32_t hc, uint32_t d) {
    TT_FATAL(
        hidden.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "hidden_streams must be WIDTH_SHARDED");
    const auto& shard_spec = hidden.shard_spec();
    TT_FATAL(shard_spec.has_value(), "hidden_streams must have a shard specification");
    const uint32_t num_cores = shard_spec->grid.num_cores();
    TT_FATAL(num_cores >= 1, "hidden_streams must use at least one core");
    TT_FATAL(
        shard_spec->grid.contains(CoreCoord{0, 0}),
        "hidden_streams shard grid must include logical core (0,0); fused_w is broadcast from there");
    TT_FATAL(d % num_cores == 0, "hidden size D={} must be divisible by {} shard cores", d, num_cores);
    const uint32_t shard_w = d / num_cores;
    TT_FATAL(
        shard_spec->shape[1] == shard_w,
        "hidden_streams shard width must be {}, got {}",
        shard_w,
        shard_spec->shape[1]);
    TT_FATAL(
        shard_w % tt::constants::TILE_WIDTH == 0, "hidden_streams shard width {} must be a multiple of 32", shard_w);
    if (hidden.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(hc <= 4, "ROW_MAJOR hidden_streams uses a 4x32 compute tile and requires H<=4, got {}", hc);
        TT_FATAL(
            shard_spec->shape[0] == hc,
            "ROW_MAJOR hidden_streams shard height must equal H={}, got {}",
            hc,
            shard_spec->shape[0]);
    } else {
        TT_FATAL(
            shard_spec->shape[0] % tt::constants::TILE_HEIGHT == 0,
            "TILE hidden_streams shard height {} must be a multiple of 32",
            shard_spec->shape[0]);
    }

    const auto device_grid = hidden.device()->compute_with_storage_grid_size();
    TT_FATAL(
        static_cast<uint32_t>(device_grid.x * device_grid.y) >= num_cores + 2,
        "single-user hyperconnection needs two free cores for post/comb besides the {} collapse cores",
        num_cores);
}

void validate_single_user_tensors(const FusedSingleUserParams& attributes, const FusedSingleUserInputs& tensor_args) {
    const auto& fused_w = tensor_args.fused_w;
    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& comb_bias = tensor_args.comb_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;

    TT_FATAL(fused_w.storage_type() == StorageType::DEVICE, "fused_w must be on device");
    TT_FATAL(
        fused_w.layout() == Layout::TILE || fused_w.layout() == Layout::ROW_MAJOR,
        "fused_w must be TILE or ROW_MAJOR, got {}",
        fused_w.layout());
    TT_FATAL(fused_w.dtype() == DataType::BFLOAT16, "fused_w must be BFLOAT16");
    if (fused_w.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            fused_w.logical_shape()[-2] == 1,
            "ROW_MAJOR fused_w is the decode 1x32 path and requires a single row, got height {}",
            fused_w.logical_shape()[-2]);
    }
    for (const auto* tensor : {&pre_bias, &post_bias, &comb_bias}) {
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "all fused hyperconnection inputs must be on device");
        TT_FATAL(tensor->layout() == Layout::TILE, "pre/post/comb bias must use TILE layout");
        TT_FATAL(tensor->dtype() == DataType::BFLOAT16, "all fused hyperconnection inputs must be BFLOAT16");
    }
    TT_FATAL(hidden_streams.storage_type() == StorageType::DEVICE, "hidden_streams must be on device");
    TT_FATAL(
        hidden_streams.layout() == Layout::TILE || hidden_streams.layout() == Layout::ROW_MAJOR,
        "hidden_streams must be TILE or ROW_MAJOR, got {}",
        hidden_streams.layout());
    TT_FATAL(hidden_streams.dtype() == DataType::BFLOAT16, "hidden_streams must be BFLOAT16");

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

    validate_fused_w_shard(fused_w);
    validate_hidden_shard(hidden_streams, hc, d);
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
    const auto make_post_comb_layout = [&](const MemoryConfig& memory_config) {
        // mix_streams requires TILE 32x32 post/comb regardless of fused_w's layout.
        return tt::tt_metal::TensorLayout(
            tensor_args.fused_w.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), memory_config);
    };
    const auto make_collapsed_layout = [&](const MemoryConfig& memory_config) {
        // Decode packs collapsed as 1x32 faces onto hidden's cores (shard height 1), which
        // is physically RM. TILE hidden + TILE fused_w keeps a 32x32 collapsed tile.
        const bool collapsed_rm = tensor_args.fused_w.layout() == Layout::ROW_MAJOR ||
                                  tensor_args.hidden_streams.layout() == Layout::ROW_MAJOR;
        const auto layout = collapsed_rm ? Layout::ROW_MAJOR : tensor_args.hidden_streams.layout();
        return tt::tt_metal::TensorLayout(
            tensor_args.hidden_streams.dtype(), tt::tt_metal::PageConfig(layout), memory_config);
    };
    const auto& hidden_shape = tensor_args.hidden_streams.logical_shape();
    const uint32_t d = static_cast<uint32_t>(hidden_shape[3]);
    return {
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, attributes.num_streams, 1}),
            make_post_comb_layout(attributes.post_comb_output_mem_config)),
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, attributes.num_streams, attributes.num_streams}),
            make_post_comb_layout(attributes.post_comb_output_mem_config)),
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, 1, d}), make_collapsed_layout(attributes.collapsed_output_mem_config))};
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
    MemoryConfig collapsed_output_mem_config = memory_config.value_or(hidden_streams.memory_config());
    const bool collapsed_rm = fused_w.layout() == Layout::ROW_MAJOR || hidden_streams.layout() == Layout::ROW_MAJOR;
    if (!memory_config.has_value() && collapsed_rm) {
        // Pack the 1x32 collapse row onto hidden's cores with shard height 1 (RM).
        if (hidden_streams.shard_spec().has_value()) {
            auto shard = hidden_streams.shard_spec().value();
            shard.shape[0] = 1;
            collapsed_output_mem_config = MemoryConfig(
                hidden_streams.memory_config().memory_layout(), hidden_streams.memory_config().buffer_type(), shard);
        } else {
            collapsed_output_mem_config =
                MemoryConfig(TensorMemoryLayout::INTERLEAVED, hidden_streams.memory_config().buffer_type());
        }
    }
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
