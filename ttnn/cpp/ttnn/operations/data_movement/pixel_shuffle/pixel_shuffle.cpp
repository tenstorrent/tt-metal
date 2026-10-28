// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pixel_shuffle.hpp"
#include "device/pixel_shuffle_device_op.hpp"
#include "ttnn/operations/core/core.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn {

Tensor pixel_shuffle(
    const Tensor& input_tensor,
    uint32_t upscale_factor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Layout>& output_layout,
    PixelUnshuffleChannelOrder channel_order,
    bool channels_last,
    const std::optional<uint32_t>& padded_channels) {
    const uint32_t r = upscale_factor;
    TT_FATAL(
        input_tensor.logical_shape().rank() == 4,
        "pixel_shuffle: input must be 4D ([N, C*r^2, H, W], or [N, H, W, C*r^2] with channels_last), got rank {}.",
        input_tensor.logical_shape().rank());
    TT_FATAL(r > 0, "pixel_shuffle: upscale_factor must be positive, got {}.", r);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "pixel_shuffle: input must be on device.");

    const auto& shape = input_tensor.logical_shape();
    const uint32_t c_in = channels_last ? shape[3] : shape[1];
    TT_FATAL(
        c_in % (r * r) == 0, "pixel_shuffle: input channels {} must be divisible by upscale_factor^2 = {}.", c_in, r * r);
    const uint32_t C = c_in / (r * r);

    const auto dtype = input_tensor.dtype();
    TT_FATAL(
        dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32 || dtype == DataType::UINT16 ||
            dtype == DataType::INT32,
        "pixel_shuffle: unsupported dtype {}. Supported: BFLOAT16, FLOAT32, UINT16, INT32.",
        dtype);

    // ── channels_last: NHWC in (height-sharded L1) -> NHWC out (height-sharded L1, padded channels) ──
    if (channels_last) {
        TT_FATAL(
            input_tensor.layout() == Layout::ROW_MAJOR,
            "pixel_shuffle: channels_last input is the ROW_MAJOR NHWC activation ([N, H, W, C*r^2]); untilize first.");
        TT_FATAL(
            !(output_layout.has_value() && output_layout.value() == Layout::TILE),
            "pixel_shuffle: channels_last output is ROW_MAJOR (the row-major sharded activation conv2d consumes "
            "directly); do not request TILE.");
        TT_FATAL(
            memory_config.has_value() && memory_config->is_sharded() &&
                memory_config->memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
                memory_config->buffer_type() == BufferType::L1,
            "pixel_shuffle: channels_last needs a HEIGHT_SHARDED L1 memory_config whose shard is "
            "[pixels_per_core, padded_channels].");
        const uint32_t datum = input_tensor.element_size();
        const uint32_t align_elems = tt::tt_metal::hal::get_l1_alignment() / datum;
        const uint32_t cp = padded_channels.value_or(tt::round_up(C, align_elems));
        return ttnn::prim::pixel_shuffle(input_tensor, r, memory_config.value(), channel_order, true, cp);
    }

    TT_FATAL(!padded_channels.has_value(), "pixel_shuffle: padded_channels is only used with channels_last.");

    // ── Normalise input: TILE -> ROW_MAJOR (the kernel reads one W-element row per page) ──
    Tensor processed = input_tensor;
    if (processed.layout() == Layout::TILE) {
        processed = ttnn::to_layout(processed, Layout::ROW_MAJOR);
    }

    // ── Output memory config: explicit, else the input's when interleaved, else DRAM ──
    MemoryConfig out_mem_config = ttnn::DRAM_MEMORY_CONFIG;
    if (memory_config.has_value()) {
        out_mem_config = memory_config.value();
    } else if (!processed.memory_config().is_sharded()) {
        out_mem_config = processed.memory_config();
    }

    Tensor result = ttnn::prim::pixel_shuffle(processed, r, out_mem_config, channel_order);

    if (output_layout.has_value() && output_layout.value() == Layout::TILE) {
        result = ttnn::to_layout(result, Layout::TILE);
    }
    return result;
}

}  // namespace ttnn
