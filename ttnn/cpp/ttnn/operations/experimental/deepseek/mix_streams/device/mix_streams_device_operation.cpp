// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mix_streams_device_operation.hpp"

#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams {

namespace {

// The fused kernel keeps comb (and its transpose) in a single tile and folds the
// placement outer product into a second matmul against the same DST accumulator,
// which is what pins hc to one tile and D to a whole number of tiles.
struct Dims {
    uint32_t b = 0;
    uint32_t s = 0;
    uint32_t hc = 0;
    uint32_t d = 0;
};

std::optional<Dims> fusable_dims(
    const Tensor& post, const Tensor& comb, const Tensor& sublayer_out, const Tensor& streams) {
    for (const Tensor* t : {&post, &comb, &sublayer_out, &streams}) {
        if (t->storage_type() != StorageType::DEVICE || t->dtype() != DataType::BFLOAT16 ||
            t->logical_shape().rank() != 4) {
            return std::nullopt;
        }
    }
    if (post.layout() != Layout::TILE || comb.layout() != Layout::TILE) {
        return std::nullopt;
    }
    if (streams.layout() != Layout::TILE && streams.layout() != Layout::ROW_MAJOR) {
        return std::nullopt;
    }
    if (sublayer_out.layout() != Layout::ROW_MAJOR && sublayer_out.layout() != Layout::TILE) {
        return std::nullopt;
    }

    const auto& streams_shape = streams.logical_shape();
    const Dims dims{
        static_cast<uint32_t>(streams_shape[0]),
        static_cast<uint32_t>(streams_shape[1]),
        static_cast<uint32_t>(streams_shape[2]),
        static_cast<uint32_t>(streams_shape[3])};

    if (dims.hc == 0 || dims.hc > tt::constants::TILE_HEIGHT || dims.d % tt::constants::TILE_WIDTH != 0) {
        return std::nullopt;
    }
    const bool shapes_ok = post.logical_shape() == ttnn::Shape({dims.b, dims.s, dims.hc, 1}) &&
                           comb.logical_shape() == ttnn::Shape({dims.b, dims.s, dims.hc, dims.hc}) &&
                           sublayer_out.logical_shape() == ttnn::Shape({dims.b, dims.s, 1, dims.d});
    if (!shapes_ok) {
        return std::nullopt;
    }
    for (const Tensor* t : {&post, &comb}) {
        const auto& tile = t->tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return std::nullopt;
        }
    }
    if (streams.layout() == Layout::TILE) {
        const auto& tile = streams.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return std::nullopt;
        }
    }
    if (sublayer_out.layout() == Layout::TILE) {
        const auto& tile = sublayer_out.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return std::nullopt;
        }
    }
    return dims;
}

void validate_tensors(const MixStreamsParams& attributes, const MixStreamsInputs& tensor_args) {
    const auto dims = fusable_dims(tensor_args.post, tensor_args.comb, tensor_args.sublayer_out, tensor_args.streams);
    TT_FATAL(
        dims.has_value(),
        "mix_streams: inputs are not supported by the fused kernel -- expected device-resident BFLOAT16 tensors "
        "post/comb TILE 32x32, streams TILE 32x32 or ROW_MAJOR, and sublayer_out ROW_MAJOR or TILE, shaped post "
        "[B,S,hc,1], comb [B,S,hc,hc], sublayer_out [B,S,1,D], streams [B,S,hc,D] with hc <= {} and D a multiple of "
        "{}; got post {}, comb {}, sublayer_out {}, streams {}",
        tt::constants::TILE_HEIGHT,
        tt::constants::TILE_WIDTH,
        tensor_args.post.logical_shape(),
        tensor_args.comb.logical_shape(),
        tensor_args.sublayer_out.logical_shape(),
        tensor_args.streams.logical_shape());
    TT_FATAL(
        attributes.num_streams == dims->hc,
        "mix_streams: num_streams {} must match the streams stream dim {}",
        attributes.num_streams,
        dims->hc);
}

}  // namespace

constexpr uint32_t kOutputCores = 64;

tt::tt_metal::CoreRangeSet rectangular_core_range_set(uint32_t num_cores, const CoreCoord& grid) {
    uint32_t x = static_cast<uint32_t>(grid.x);
    while (x > 0 && num_cores % x != 0) {
        --x;
    }
    const uint32_t y = x == 0 ? 0 : num_cores / x;
    TT_FATAL(
        x > 0 && y > 0 && y <= static_cast<uint32_t>(grid.y),
        "mix_streams: cannot form a rectangular grid of {} cores within a {}x{} device grid",
        num_cores,
        grid.x,
        grid.y);
    return tt::tt_metal::CoreRangeSet({tt::tt_metal::CoreRange(CoreCoord(0, 0), CoreCoord(x - 1, y - 1))});
}

// WIDTH_SHARDED L1 on 64 cores along D. Requires D % 64 == 0 and a tile-aligned shard
// width so the RM writer can scatter 32-wide faces into each shard row.
std::optional<MemoryConfig> width_sharded_64_core_output(const Tensor& streams, bool untilize_out) {
    const auto& shape = streams.logical_shape();
    const uint32_t d = static_cast<uint32_t>(shape[-1]);
    const uint32_t folded_h =
        static_cast<uint32_t>(shape[0]) * static_cast<uint32_t>(shape[1]) * static_cast<uint32_t>(shape[2]);
    if (d % kOutputCores != 0) {
        return std::nullopt;
    }
    const uint32_t shard_w = d / kOutputCores;
    if (shard_w % tt::constants::TILE_WIDTH != 0) {
        return std::nullopt;
    }
    const auto grid_size = streams.device()->compute_with_storage_grid_size();
    const uint32_t shard_h =
        untilize_out ? folded_h : tt::round_up(folded_h, static_cast<uint32_t>(tt::constants::TILE_HEIGHT));
    const tt::tt_metal::ShardSpec shard_spec(
        rectangular_core_range_set(kOutputCores, grid_size),
        {shard_h, shard_w},
        tt::tt_metal::ShardOrientation::ROW_MAJOR);
    return MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec);
}

bool is_fusable(const Tensor& post, const Tensor& comb, const Tensor& sublayer_out, const Tensor& streams) {
    return fusable_dims(post, comb, sublayer_out, streams).has_value();
}

void MixStreamsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(attributes, tensor_args);
}

void MixStreamsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(attributes, tensor_args);
}

MixStreamsDeviceOperation::spec_return_value_t MixStreamsDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& streams = tensor_args.streams;
    // Decode: sublayer_out is RM while residual streams are often still TILE. Untilize dest
    // whenever either input is RM, even if untilize_out was left at its default.
    const bool untilize_out = operation_attributes.untilize_out || streams.layout() == Layout::ROW_MAJOR ||
                              tensor_args.sublayer_out.layout() == Layout::ROW_MAJOR;
    const auto page_config = untilize_out ? tt::tt_metal::PageConfig(Layout::ROW_MAJOR)
                                          : tt::tt_metal::PageConfig(Layout::TILE, streams.tensor_spec().tile());
    return tt::tt_metal::TensorSpec(
        streams.logical_shape(),
        tt::tt_metal::TensorLayout(streams.dtype(), page_config, operation_attributes.output_mem_config));
}

MixStreamsDeviceOperation::tensor_return_value_t MixStreamsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.streams.device());
}

}  // namespace ttnn::operations::experimental::deepseek::mix_streams

namespace ttnn::prim {

Tensor mix_streams(
    const Tensor& post,
    const Tensor& comb,
    const Tensor& sublayer_out,
    const Tensor& streams,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::deepseek::mix_streams::MixStreamsDeviceOperation;

    // HiFi4 / fp32 dest acc / packer-l1-acc by default, matching the ``_HIFI4`` config
    // the eager Python path passes to ttnn.matmul.
    const auto kernel_config = init_device_compute_kernel_config(
        streams.device()->arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true);

    const bool untilize_out = streams.layout() == Layout::ROW_MAJOR || sublayer_out.layout() == Layout::ROW_MAJOR;
    MemoryConfig out_mem;
    if (memory_config.has_value()) {
        out_mem = *memory_config;
    } else if (auto sharded = ttnn::operations::experimental::deepseek::mix_streams::width_sharded_64_core_output(
                   streams, untilize_out);
               sharded.has_value()) {
        out_mem = *sharded;
    } else if (untilize_out && streams.layout() != Layout::ROW_MAJOR) {
        // TILE-sharded streams pad hc to 32; RM output is the unpadded hc rows, so drop
        // the incoming shard spec when 64-core WIDTH_SHARDED does not fit D.
        out_mem = MemoryConfig{TensorMemoryLayout::INTERLEAVED, streams.memory_config().buffer_type()};
    } else {
        out_mem = streams.memory_config();
    }
    auto operation_attributes = OperationType::operation_attributes_t{
        .num_streams = static_cast<uint32_t>(streams.logical_shape()[2]),
        .output_mem_config = out_mem,
        .compute_kernel_config = kernel_config,
        .untilize_out = untilize_out,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .post = post,
        .comb = comb,
        .sublayer_out = sublayer_out,
        .streams = streams,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
