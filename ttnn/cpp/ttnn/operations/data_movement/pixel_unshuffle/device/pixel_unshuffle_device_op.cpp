// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pixel_unshuffle_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/hal.hpp>

namespace ttnn::operations::data_movement {

PixelUnshuffle::program_factory_t PixelUnshuffle::select_program_factory(
    const operation_attributes_t& op_attr, const tensor_args_t& /*tensors*/) {
    if (op_attr.channels_last) {
        return MultiCoreChannelsLast{};
    }
    return MultiCore{};
}

// channels_last output: NHWC rows of `padded_channels` elements, height-sharded in L1 so each
// core can write its own shard with local stores. Everything the kernel assumes is checked here.
static void validate_channels_last(const Tensor& input_tensor, const PixelUnshuffle::operation_attributes_t& op_attr) {
    const auto& shape = input_tensor.logical_shape();
    const uint32_t N = shape[0];
    const uint32_t C = shape[1];
    const uint32_t r = op_attr.downscale_factor;
    const uint32_t Ho = shape[2] / r;
    const uint32_t Wo = shape[3] / r;
    const uint32_t c_out = C * r * r;
    const uint32_t datum = input_tensor.element_size();
    const auto& mem = op_attr.output_mem_config;

    TT_FATAL(
        datum == 2 || datum == 4, "PixelUnshuffle(channels_last): 2- or 4-byte elements only, got {} bytes.", datum);
    TT_FATAL(
        mem.is_sharded() && mem.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            mem.buffer_type() == BufferType::L1,
        "PixelUnshuffle(channels_last): output memory_config must be HEIGHT_SHARDED in L1 (each core writes its own "
        "shard), got {}.",
        mem);
    TT_FATAL(
        op_attr.padded_channels >= c_out,
        "PixelUnshuffle(channels_last): padded_channels {} < C*r^2 = {}.",
        op_attr.padded_channels,
        c_out);
    const uint32_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    TT_FATAL(
        (op_attr.padded_channels * datum) % l1_align == 0,
        "PixelUnshuffle(channels_last): padded_channels {} x {} B must be a multiple of the L1 alignment ({} B).",
        op_attr.padded_channels,
        datum,
        l1_align);
    const auto& ss = mem.shard_spec().value();
    TT_FATAL(
        ss.shape[1] == op_attr.padded_channels,
        "PixelUnshuffle(channels_last): shard width {} must equal padded_channels {}.",
        ss.shape[1],
        op_attr.padded_channels);
    const uint64_t total_pixels = static_cast<uint64_t>(N) * Ho * Wo;
    TT_FATAL(
        static_cast<uint64_t>(ss.shape[0]) * ss.num_cores() >= total_pixels,
        "PixelUnshuffle(channels_last): shard height {} x {} cores covers fewer than the {} output pixels.",
        ss.shape[0],
        ss.num_cores(),
        total_pixels);
}

static void validate_pixel_unshuffle(const Tensor& input_tensor, uint32_t r) {
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "PixelUnshuffle: input tensor must be on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "PixelUnshuffle: input tensor must be allocated on a device buffer.");
    TT_FATAL(
        input_tensor.logical_shape().rank() == 4,
        "PixelUnshuffle: input must be a 4D tensor [N, C, H, W], got rank {}.",
        input_tensor.logical_shape().rank());
    TT_FATAL(
        input_tensor.layout() == Layout::ROW_MAJOR,
        "PixelUnshuffle: input tensor must be in ROW_MAJOR layout (untilize before calling).");
    TT_FATAL(r > 0, "PixelUnshuffle: downscale_factor must be positive, got {}.", r);

    const auto& shape = input_tensor.logical_shape();
    const uint32_t H = shape[2];
    const uint32_t W = shape[3];
    TT_FATAL(W > 0, "PixelUnshuffle: input width W must be > 0.");
    TT_FATAL(H % r == 0, "PixelUnshuffle: input height {} must be divisible by downscale_factor {}.", H, r);
    TT_FATAL(W % r == 0, "PixelUnshuffle: input width {} must be divisible by downscale_factor {}.", W, r);
    // Sharded input is supported: TensorAccessor resolves any page_id to the correct
    // (core, offset) via NOC reads, whether the buffer is interleaved or sharded.
}

void PixelUnshuffle::validate_on_program_cache_miss(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_pixel_unshuffle(tensors.input_tensor, op_attr.downscale_factor);
    if (op_attr.channels_last) {
        validate_channels_last(tensors.input_tensor, op_attr);
    }
}

void PixelUnshuffle::validate_on_program_cache_hit(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_pixel_unshuffle(tensors.input_tensor, op_attr.downscale_factor);
    if (op_attr.channels_last) {
        validate_channels_last(tensors.input_tensor, op_attr);
    }
}

PixelUnshuffle::spec_return_value_t PixelUnshuffle::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    const auto& input = tensors.input_tensor;
    const auto& shape = input.logical_shape();
    const uint32_t N = shape[0];
    const uint32_t C = shape[1];
    const uint32_t H = shape[2];
    const uint32_t W = shape[3];
    const uint32_t r = op_attr.downscale_factor;

    // NCHW by default; NHWC with zero-padded channels for channels_last.
    const ttnn::Shape output_shape = op_attr.channels_last ? ttnn::Shape({N, H / r, W / r, op_attr.padded_channels})
                                                           : ttnn::Shape({N, C * r * r, H / r, W / r});

    // Preserve input dtype for 4-byte types (FLOAT32, INT32) and 2-byte UINT16.
    // All other dtypes (BFLOAT16, etc.) pass through unchanged.
    const auto in_dt = input.dtype();
    tt::tt_metal::DataType output_dtype =
        (in_dt == tt::tt_metal::DataType::FLOAT32 || in_dt == tt::tt_metal::DataType::INT32 ||
         in_dt == tt::tt_metal::DataType::UINT16)
            ? in_dt
            : tt::tt_metal::DataType::BFLOAT16;

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), op_attr.output_mem_config));
}

PixelUnshuffle::tensor_return_value_t PixelUnshuffle::create_output_tensors(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(op_attr, tensors), tensors.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PixelUnshuffle::tensor_return_value_t pixel_unshuffle(
    const ttnn::Tensor& input_tensor,
    uint32_t downscale_factor,
    const MemoryConfig& output_mem_config,
    ttnn::PixelUnshuffleChannelOrder channel_order,
    bool channels_last,
    uint32_t padded_channels) {
    using OpType = ttnn::operations::data_movement::PixelUnshuffle;
    auto op_attr = OpType::operation_attributes_t{
        .downscale_factor = downscale_factor,
        .output_mem_config = output_mem_config,
        .channel_order = channel_order,
        .channels_last = channels_last,
        .padded_channels = padded_channels};
    auto tensor_args = OpType::tensor_args_t{.input_tensor = input_tensor};
    return ttnn::device_operation::launch<OpType>(op_attr, tensor_args);
}
}  // namespace ttnn::prim
