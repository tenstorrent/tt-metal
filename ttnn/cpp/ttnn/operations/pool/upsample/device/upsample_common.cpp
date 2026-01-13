// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_common.hpp"

#include <cmath>

namespace ttnn::operations::pool::upsample {

bool is_integer_scale(float scale) { return scale == std::floor(scale); }

uint32_t compute_num_cores_nhw(const tt::tt_metal::ShardSpec& shard_spec, tt::tt_metal::TensorMemoryLayout mem_layout) {
    if (mem_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        return shard_spec.grid.num_cores();
    }
    if (mem_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        return 1;
    }
    // BLOCK_SHARDED
    const auto grid_size = shard_spec.grid.bounding_box().grid_size();
    return (shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR) ? grid_size.y : grid_size.x;
}

UpsamplePath select_upsample_path(const Tensor& input, float scale_h, float scale_w, const std::string& mode) {
    const bool scales_are_integer = is_integer_scale(scale_h) && is_integer_scale(scale_w);
    const auto& mem_config = input.memory_config();
    const tt::tt_metal::Layout layout = input.layout();

    // Bilinear: requires integer scales, always uses integer path
    if (mode == "bilinear") {
        return scales_are_integer ? UpsamplePath::INTEGER_OPTIMIZED : UpsamplePath::UNSUPPORTED;
    }

    // Nearest mode
    if (mode == "nearest") {
        // ND sharded → always float path (integer path doesn't support ND sharding)
        if (mem_config.is_sharded() && mem_config.created_with_nd_shard_spec()) {
            return layout == tt::tt_metal::Layout::ROW_MAJOR ? UpsamplePath::FLOAT_GENERAL : UpsamplePath::UNSUPPORTED;
        }

        // Integer scales + supported config → integer path
        if (scales_are_integer) {
            // TILE layout only supported for interleaved
            if (layout == tt::tt_metal::Layout::TILE &&
                mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
                return UpsamplePath::INTEGER_OPTIMIZED;
            }
            // ROW_MAJOR supports interleaved, height_sharded, block_sharded
            if (layout == tt::tt_metal::Layout::ROW_MAJOR) {
                const auto mem_layout = mem_config.memory_layout();
                if (mem_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
                    mem_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
                    mem_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
                    return UpsamplePath::INTEGER_OPTIMIZED;
                }
            }
        }

        // Float path fallback (ROW_MAJOR only)
        return layout == tt::tt_metal::Layout::ROW_MAJOR ? UpsamplePath::FLOAT_GENERAL : UpsamplePath::UNSUPPORTED;
    }

    return UpsamplePath::UNSUPPORTED;
}

tt::tt_metal::MemoryConfig compute_nd_output_mem_config(
    const tt::tt_metal::MemoryConfig& input_mem_config, float scale_h, float scale_w) {
    const auto& in_spec = input_mem_config.nd_shard_spec().value();
    const ttnn::Shape output_shard_shape(std::array<uint32_t, 4>{
        in_spec.shard_shape[0],
        static_cast<uint32_t>(std::ceil(in_spec.shard_shape[1] * scale_h)),
        static_cast<uint32_t>(std::ceil(in_spec.shard_shape[2] * scale_w)),
        in_spec.shard_shape[3]});
    return tt::tt_metal::MemoryConfig(
        input_mem_config.buffer_type(),
        tt::tt_metal::NdShardSpec{
            output_shard_shape, in_spec.grid, in_spec.orientation, in_spec.shard_distribution_strategy});
}

tt::tt_metal::MemoryConfig compute_integer_output_mem_config(
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const Tensor& input,
    const std::string& mode,
    float scale_h,
    float scale_w,
    uint32_t out_n,
    uint32_t out_h,
    uint32_t out_w) {
    auto shard_spec = output_mem_config.shard_spec().value();
    if (mode == "bilinear") {
        // Bilinear: calculate to handle non-exact work distribution (input is haloed)
        const uint32_t total = out_n * out_h * out_w;
        const uint32_t padded = tt::round_up(total, shard_spec.num_cores());
        shard_spec.shape = {padded / shard_spec.num_cores(), input.shard_spec()->shape[1]};
    } else {
        // Nearest: output shard is input shard * scale factors
        shard_spec.shape = {input.shard_spec()->shape[0] * scale_h * scale_w, input.shard_spec()->shape[1]};
    }
    return output_mem_config.with_shard_spec(shard_spec);
}

tt::tt_metal::MemoryConfig compute_float_output_mem_config(
    const tt::tt_metal::MemoryConfig& input_mem_config, uint32_t out_n, uint32_t out_h, uint32_t out_w) {
    const auto& in_spec = input_mem_config.shard_spec().value();
    const uint32_t num_cores_nhw = compute_num_cores_nhw(in_spec, input_mem_config.memory_layout());
    const uint32_t output_nhw = out_n * out_h * out_w;
    tt::tt_metal::ShardSpec out_spec(
        in_spec.grid, {tt::div_up(output_nhw, num_cores_nhw), in_spec.shape[1]}, in_spec.orientation);
    return tt::tt_metal::MemoryConfig(input_mem_config.memory_layout(), input_mem_config.buffer_type(), out_spec);
}

std::string generate_unsupported_config_message(
    const Tensor& input, float scale_h, float scale_w, const std::string& mode) {
    std::string msg = "Unsupported upsample configuration:";

    if (mode == "bilinear" && !(is_integer_scale(scale_h) && is_integer_scale(scale_w))) {
        msg += " bilinear mode requires integer scale factors (got " + std::to_string(scale_h) + ", " +
               std::to_string(scale_w) + ")";
        return msg;
    }

    if (mode != "nearest" && mode != "bilinear") {
        msg += " mode must be 'nearest' or 'bilinear' (got '" + mode + "')";
        return msg;
    }

    const auto layout = input.layout();
    const auto mem_layout = input.memory_config().memory_layout();

    msg += "\n  Mode: " + mode;
    msg += "\n  Scales: " + std::to_string(scale_h) + " x " + std::to_string(scale_w);
    msg += "\n  Layout: " + std::string(layout == tt::tt_metal::Layout::TILE ? "TILE" : "ROW_MAJOR");
    msg += "\n  Memory: ";
    switch (mem_layout) {
        case tt::tt_metal::TensorMemoryLayout::INTERLEAVED: msg += "INTERLEAVED"; break;
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: msg += "HEIGHT_SHARDED"; break;
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: msg += "WIDTH_SHARDED"; break;
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: msg += "BLOCK_SHARDED"; break;
        default: msg += "OTHER";
    }

    msg += "\n\nSupported configurations:";
    msg += "\n  - Integer scales + nearest + ROW_MAJOR + (INTERLEAVED|HEIGHT_SHARDED|BLOCK_SHARDED)";
    msg += "\n  - Integer scales + nearest + TILE + INTERLEAVED";
    msg += "\n  - Float scales + nearest + ROW_MAJOR + any memory layout";
    msg += "\n  - Integer scales + bilinear + HEIGHT_SHARDED or INTERLEAVED)";

    return msg;
}

}  // namespace ttnn::operations::pool::upsample
