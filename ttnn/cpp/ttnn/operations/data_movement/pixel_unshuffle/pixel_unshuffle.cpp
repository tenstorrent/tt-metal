// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pixel_unshuffle.hpp"
#include "device/pixel_unshuffle_device_op.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

Tensor pixel_unshuffle(
    const Tensor& input_tensor,
    uint32_t downscale_factor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Layout>& output_layout,
    PixelUnshuffleChannelOrder channel_order) {
    // ── Validation ────────────────────────────────────────────────────────────

    TT_FATAL(
        input_tensor.logical_shape().rank() == 4,
        "pixel_unshuffle: input must be 4D [N,C,H,W], got rank {}.",
        input_tensor.logical_shape().rank());
    TT_FATAL(downscale_factor > 0, "pixel_unshuffle: downscale_factor must be positive, got {}.", downscale_factor);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "pixel_unshuffle: input must be on device.");

    const auto& shape = input_tensor.logical_shape();
    TT_FATAL(
        shape[2] % downscale_factor == 0,
        "pixel_unshuffle: H={} must be divisible by downscale_factor={}.",
        shape[2],
        downscale_factor);
    TT_FATAL(
        shape[3] % downscale_factor == 0,
        "pixel_unshuffle: W={} must be divisible by downscale_factor={}.",
        shape[3],
        downscale_factor);

    const auto dtype = input_tensor.dtype();
    TT_FATAL(
        dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32 || dtype == DataType::UINT16 ||
            dtype == DataType::INT32,
        "pixel_unshuffle: unsupported dtype {}. Supported: BFLOAT16, FLOAT32, UINT16, INT32.",
        dtype);

    // ── Normalise input: TILE → ROW_MAJOR ────────────────────────────────────
    // The dedicated kernel operates on ROW_MAJOR pages (one W-element row = one page).
    // Sharded input is supported natively: TensorAccessor encodes the full shard routing
    // table (core grid, shard shape, bank addresses) and resolves any page_id to the
    // correct (core, offset) via NOC reads, whether the buffer is interleaved or sharded.
    Tensor processed = input_tensor;
    if (processed.layout() == Layout::TILE) {
        processed = ttnn::to_layout(processed, Layout::ROW_MAJOR);
    }

    // ── Output memory config ──────────────────────────────────────────────────
    // If the caller provides an explicit memory config, use it.
    // If the (normalised) input is interleaved, inherit it for output.
    // If the input is sharded, do NOT inherit: the output has a different shape
    // (C_out = C*r²) so the shard spec would be invalid — default to DRAM interleaved.
    MemoryConfig out_mem_config = ttnn::DRAM_MEMORY_CONFIG;
    if (memory_config.has_value()) {
        out_mem_config = memory_config.value();
    } else if (!processed.memory_config().is_sharded()) {
        out_mem_config = processed.memory_config();
    }

    // ── Run kernel ────────────────────────────────────────────────────────────
    Tensor result = ttnn::prim::pixel_unshuffle(processed, downscale_factor, out_mem_config, channel_order);

    // ── Optional: tilize output ───────────────────────────────────────────────
    // The kernel always produces ROW_MAJOR.  If the caller requests TILE output,
    // tilize here so the result is immediately usable by downstream TILE ops.
    if (output_layout.has_value() && output_layout.value() == Layout::TILE) {
        result = ttnn::to_layout(result, Layout::TILE);
    }

    return result;
}

}  // namespace ttnn
