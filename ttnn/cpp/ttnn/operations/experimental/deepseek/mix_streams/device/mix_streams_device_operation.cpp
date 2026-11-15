// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mix_streams_device_operation.hpp"

#include <tt-metalium/constants.hpp>

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
    for (const Tensor* t : {&post, &comb, &streams}) {
        if (t->layout() != Layout::TILE) {
            continue;
        }
        const auto& tile = t->tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return std::nullopt;
        }
    }
    // ROW_MAJOR streams are read 32 columns at a time out of their row pages, in 32 B halves -- an
    // alignment L1 honours and DRAM (64 B) does not.
    if (streams.layout() == Layout::ROW_MAJOR) {
        if (streams.memory_config().buffer_type() != BufferType::L1) {
            return std::nullopt;
        }
        const uint32_t page_elems =
            streams.shard_spec().has_value() ? static_cast<uint32_t>(streams.shard_spec()->shape[1]) : dims.d;
        if (page_elems % tt::constants::TILE_WIDTH != 0 || dims.d % page_elems != 0) {
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

// A block of up to h rows of one session: streams [1, 1, hc * h, D] in (h x 32) tiles (row s * h + j = stream s of
// block row j), sublayer_out [1, 1, h, D] in the same tiles, post [1, n, hc, 1] and comb [1, n, hc, hc] TILE 32x32
// with block row j at dim 1 (n <= h rows; rows n..h-1 come out 0).
void validate_block(const MixStreamsParams& attributes, const MixStreamsInputs& tensor_args) {
    const auto& [post, comb, sub, streams] = tensor_args;
    const uint32_t h = block_height(streams);
    TT_FATAL(h == 1 || h == 2 || h == 4 || h == 8, "mix_streams: block tile height must be 1, 2, 4 or 8, got {}", h);
    for (const Tensor* t : {&post, &comb, &sub, &streams}) {
        TT_FATAL(
            t->storage_type() == StorageType::DEVICE && t->dtype() == DataType::BFLOAT16 &&
                t->layout() == Layout::TILE && t->logical_shape().rank() == 4,
            "mix_streams: block inputs must be rank-4 BFLOAT16 TILE device tensors");
    }
    const uint32_t hc = attributes.num_streams;
    const uint32_t n = post.logical_shape()[1];
    const uint32_t d = streams.logical_shape()[3];
    TT_FATAL(hc >= 1 && hc * h <= tt::constants::TILE_HEIGHT, "mix_streams: hc * h = {} must be <= 32", hc * h);
    TT_FATAL(n >= 1 && n <= h, "mix_streams: block rows {} must be in [1, {}]", n, h);
    TT_FATAL(d % tt::constants::TILE_WIDTH == 0, "mix_streams: D {} must be a multiple of 32", d);
    TT_FATAL(
        streams.logical_shape() == ttnn::Shape({1, 1, hc * h, d}) &&
            streams.tensor_spec().tile().get_width() == tt::constants::TILE_WIDTH,
        "mix_streams: block streams must be [1, 1, {}, {}] in {}x32 tiles, got {}",
        hc * h,
        d,
        h,
        streams.logical_shape());
    TT_FATAL(
        sub.padded_shape() == ttnn::Shape({1, 1, h, d}) && sub.tensor_spec().tile().get_height() == h &&
            sub.tensor_spec().tile().get_width() == tt::constants::TILE_WIDTH,
        "mix_streams: block sublayer_out must be [1, 1, {}, {}] in the streams' {}x32 tiles, got {} ({})",
        h,
        d,
        h,
        sub.logical_shape(),
        sub.tensor_spec().tile().get_tile_shape());
    TT_FATAL(
        post.logical_shape() == ttnn::Shape({1, n, hc, 1}) && comb.logical_shape() == ttnn::Shape({1, n, hc, hc}),
        "mix_streams: block post / comb must be [1, {}, {}, 1] / [1, {}, {}, {}], got {} / {}",
        n,
        hc,
        n,
        hc,
        hc,
        post.logical_shape(),
        comb.logical_shape());
    for (const Tensor* t : {&post, &comb}) {
        const auto& tile = t->tensor_spec().tile();
        TT_FATAL(
            tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
            "mix_streams: block post / comb must use 32x32 tiles");
    }
}

void validate_tensors(const MixStreamsParams& attributes, const MixStreamsInputs& tensor_args) {
    if (block_height(tensor_args.streams) > 0) {
        validate_block(attributes, tensor_args);
        return;
    }
    const auto dims = fusable_dims(tensor_args.post, tensor_args.comb, tensor_args.sublayer_out, tensor_args.streams);
    TT_FATAL(
        dims.has_value(),
        "mix_streams: inputs are not supported by the fused kernel -- expected device-resident BFLOAT16 tensors "
        "post/comb TILE 32x32, streams TILE 32x32 or ROW_MAJOR (row pages a multiple of 32 wide) and "
        "sublayer_out ROW_MAJOR or TILE, shaped post [B,S,hc,1], comb [B,S,hc,hc], "
        "sublayer_out [B,S,1,D], streams [B,S,hc,D] with hc <= {} and D a multiple of {}; "
        "got post {}, comb {}, sublayer_out {}, streams {}",
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

uint32_t block_height(const Tensor& streams) {
    if (streams.layout() != Layout::TILE) {
        return 0;
    }
    const uint32_t h = streams.tensor_spec().tile().get_height();
    return h < tt::constants::TILE_HEIGHT ? h : 0;
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
    // The output takes the streams' layout, so a step's streams keep one format throughout.
    const auto page_config = streams.layout() == Layout::ROW_MAJOR
                                 ? tt::tt_metal::PageConfig(Layout::ROW_MAJOR)
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
        /*default_fidelity=*/tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true);

    auto operation_attributes = OperationType::operation_attributes_t{
        // A block's streams stack hc streams of h rows each; hc is post's stream dim.
        .num_streams = ttnn::operations::experimental::deepseek::mix_streams::block_height(streams) > 0
                           ? static_cast<uint32_t>(post.logical_shape()[2])
                           : static_cast<uint32_t>(streams.logical_shape()[2]),
        .output_mem_config = memory_config.value_or(streams.memory_config()),
        .compute_kernel_config = kernel_config,
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
