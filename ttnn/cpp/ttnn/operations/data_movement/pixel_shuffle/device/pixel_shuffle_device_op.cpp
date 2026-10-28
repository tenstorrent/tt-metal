// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pixel_shuffle_device_op.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/hal.hpp>

namespace ttnn::operations::data_movement {

PixelShuffle::program_factory_t PixelShuffle::select_program_factory(
    const operation_attributes_t& op_attr, const tensor_args_t& /*tensors*/) {
    if (op_attr.channels_last) {
        return MultiCoreChannelsLast{};
    }
    return MultiCore{};
}

static void validate_common(const Tensor& in, uint32_t r) {
    TT_FATAL(in.storage_type() == StorageType::DEVICE, "PixelShuffle: input tensor must be on device.");
    TT_FATAL(in.buffer() != nullptr, "PixelShuffle: input tensor must be allocated on a device buffer.");
    TT_FATAL(in.logical_shape().rank() == 4, "PixelShuffle: input must be 4D, got rank {}.", in.logical_shape().rank());
    TT_FATAL(in.layout() == Layout::ROW_MAJOR, "PixelShuffle: input must be ROW_MAJOR (untilize before calling).");
    TT_FATAL(r > 0, "PixelShuffle: upscale_factor must be positive, got {}.", r);
}

static void validate_nchw(const Tensor& in, const PixelShuffle::operation_attributes_t& a) {
    const auto& s = in.logical_shape();
    const uint32_t r = a.upscale_factor;
    TT_FATAL(s[1] % (r * r) == 0, "PixelShuffle: input channels {} must be divisible by r^2 = {}.", s[1], r * r);
    TT_FATAL(s[3] > 0, "PixelShuffle: input width must be > 0.");
}

static void validate_channels_last(const Tensor& in, const PixelShuffle::operation_attributes_t& a) {
    const auto& s = in.logical_shape();
    const uint32_t r = a.upscale_factor;
    const uint32_t N = s[0];
    const uint32_t H = s[1];
    const uint32_t W = s[2];
    const uint32_t c_in = s[3];
    TT_FATAL(c_in % (r * r) == 0, "PixelShuffle(channels_last): input channels {} must be divisible by r^2 = {}.", c_in, r * r);
    const uint32_t C = c_in / (r * r);
    const uint32_t datum = in.element_size();
    TT_FATAL(datum == 2 || datum == 4, "PixelShuffle(channels_last): 2- or 4-byte elements only, got {} bytes.", datum);
    TT_FATAL(
        in.memory_config().is_sharded() && in.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            in.memory_config().buffer_type() == BufferType::L1,
        "PixelShuffle(channels_last): input must be HEIGHT_SHARDED in L1, got {}.",
        in.memory_config());
    TT_FATAL(in.padded_shape()[3] == c_in, "PixelShuffle(channels_last): input channels must not be padded ({} vs {}).", in.padded_shape()[3], c_in);
    const auto& mem = a.output_mem_config;
    TT_FATAL(
        mem.is_sharded() && mem.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED && mem.buffer_type() == BufferType::L1,
        "PixelShuffle(channels_last): output memory_config must be HEIGHT_SHARDED in L1, got {}.",
        mem);
    TT_FATAL(a.padded_channels >= C, "PixelShuffle(channels_last): padded_channels {} < C = {}.", a.padded_channels, C);
    const uint32_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    TT_FATAL(
        (a.padded_channels * datum) % l1_align == 0,
        "PixelShuffle(channels_last): padded_channels {} x {} B must be a multiple of the L1 alignment ({} B).",
        a.padded_channels, datum, l1_align);
    const auto& ss = mem.shard_spec().value();
    TT_FATAL(ss.shape[1] == a.padded_channels, "PixelShuffle(channels_last): output shard width {} must equal padded_channels {}.", ss.shape[1], a.padded_channels);
    const uint64_t out_rows = static_cast<uint64_t>(N) * H * W * r * r;
    TT_FATAL(
        static_cast<uint64_t>(ss.shape[0]) * ss.num_cores() >= out_rows,
        "PixelShuffle(channels_last): output shards cover {} rows, need {}.", static_cast<uint64_t>(ss.shape[0]) * ss.num_cores(), out_rows);
    const bool fast = a.channel_order == ttnn::PixelUnshuffleChannelOrder::SPATIAL_MAJOR && a.padded_channels == C;
    if (fast) {
        TT_FATAL(ss.shape[0] % r == 0, "PixelShuffle(channels_last): output shard height {} must be a multiple of r = {} (one write per kernel row).", ss.shape[0], r);
    }
}

void PixelShuffle::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& t) {
    validate_common(t.input_tensor, a.upscale_factor);
    if (a.channels_last) {
        validate_channels_last(t.input_tensor, a);
    } else {
        validate_nchw(t.input_tensor, a);
    }
}

void PixelShuffle::validate_on_program_cache_hit(const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_miss(a, t);
}

PixelShuffle::spec_return_value_t PixelShuffle::compute_output_specs(const operation_attributes_t& a, const tensor_args_t& t) {
    const auto& in = t.input_tensor;
    const auto& s = in.logical_shape();
    const uint32_t r = a.upscale_factor;
    ttnn::Shape out_shape = a.channels_last ? ttnn::Shape({s[0], s[1] * r, s[2] * r, a.padded_channels})
                                            : ttnn::Shape({s[0], s[1] / (r * r), s[2] * r, s[3] * r});
    const auto in_dt = in.dtype();
    const tt::tt_metal::DataType out_dt =
        (in_dt == tt::tt_metal::DataType::FLOAT32 || in_dt == tt::tt_metal::DataType::INT32 || in_dt == tt::tt_metal::DataType::UINT16)
            ? in_dt
            : tt::tt_metal::DataType::BFLOAT16;
    return tt::tt_metal::TensorSpec(
        out_shape, tt::tt_metal::TensorLayout(out_dt, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), a.output_mem_config));
}

PixelShuffle::tensor_return_value_t PixelShuffle::create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(a, t), t.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PixelShuffle::tensor_return_value_t pixel_shuffle(
    const ttnn::Tensor& input_tensor,
    uint32_t upscale_factor,
    const MemoryConfig& output_mem_config,
    ttnn::PixelUnshuffleChannelOrder channel_order,
    bool channels_last,
    uint32_t padded_channels) {
    using OpType = ttnn::operations::data_movement::PixelShuffle;
    return ttnn::device_operation::launch<OpType>(
        OpType::operation_attributes_t{
            .upscale_factor = upscale_factor,
            .output_mem_config = output_mem_config,
            .channel_order = channel_order,
            .channels_last = channels_last,
            .padded_channels = padded_channels},
        OpType::tensor_args_t{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim
