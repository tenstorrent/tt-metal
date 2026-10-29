// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dequantize_nvfp4_device_operation.hpp"

#include "ttnn/tensor/tensor_ops.hpp"

#include <cmath>
#include <limits>
#include <utility>
#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::experimental::prim {
void Nvfp4DeviceOperation::validate_on_program_cache_miss(const Nvfp4Params& args, const Nvfp4Inputs& inputs) {
    using namespace tt::tt_metal;
    for (const auto* tensor : {&inputs.packed, &inputs.scales}) {
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "NVFP4 inputs must be device tensors");
        TT_FATAL(tensor->buffer() != nullptr, "NVFP4 input buffer is null");
        TT_FATAL(!tensor->is_sharded(), "NVFP4 requires interleaved inputs");
        TT_FATAL(
            tensor->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "NVFP4 requires interleaved inputs");
        TT_FATAL(tensor->logical_shape() == tensor->padded_shape(), "NVFP4 inputs must have no padding");
    }
    const auto& packed = inputs.packed;
    TT_FATAL(packed.device() == inputs.scales.device(), "NVFP4 inputs must be on the same device");
    TT_FATAL(packed.device()->arch() == tt::ARCH::BLACKHOLE, "NVFP4 SFPU dequantization requires Blackhole");
    TT_FATAL(
        packed.dtype() == DataType::UINT8 || packed.dtype() == DataType::UINT16,
        "NVFP4 packed data must be UINT8 or UINT16");
    TT_FATAL(packed.layout() == Layout::ROW_MAJOR || packed.layout() == Layout::TILE, "Unsupported packed layout");
    const auto& shape = packed.logical_shape();
    TT_FATAL(shape.size() >= 2 && shape.size() <= 4, "NVFP4 packed data must have rank 2 to 4");
    TT_FATAL(
        shape[-2] > 0 && shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] > 0 &&
            shape[-1] % tt::constants::TILE_WIDTH == 0,
        "NVFP4 packed height and width must be positive multiples of 32");
    TT_FATAL(packed.tensor_spec().tile() == Tile({32, 32}), "NVFP4 requires standard 32x32 tiles");
    const uint32_t planes = packed.element_size() * 2;
    const uint32_t packed_tiles = packed.logical_volume() / tt::constants::TILE_HW;
    TT_FATAL(
        (inputs.scales.dtype() == (args.experimental_fast ? DataType::BFLOAT16 : DataType::UINT8)) &&
            inputs.scales.layout() == Layout::ROW_MAJOR,
        "NVFP4 scales must be row-major UINT8 codes, or BFLOAT16 prepared scales for experimental_fast");
    TT_FATAL(
        inputs.scales.logical_shape() == Shape({packed_tiles, planes * 64}),
        "NVFP4 scales must have shape [packed_tile_count, 64 * values_per_word]");
    TT_FATAL(
        args.scale_format == operations::experimental::Nvfp4ScaleFormat::E4M3 ||
            args.scale_format == operations::experimental::Nvfp4ScaleFormat::E2M1,
        "Unsupported NVFP4 scale format");
    TT_FATAL(
        !args.transpose_output || (args.experimental_fast && args.output_layout == Layout::TILE),
        "transpose_output requires experimental_fast=True and TILE output");
    if (args.experimental_fast) {
        TT_FATAL(
            packed.dtype() == DataType::UINT16 && !args.fp32_dest_acc_en && args.dtype == DataType::BFLOAT16,
            "experimental_fast requires UINT16 input, BFLOAT16 output, and BF16 DEST");
        TT_FATAL(
            args.global_scale <= 0x1.fep127f, "experimental_fast tensor scale must remain finite when rounded to BF16");
    }
    TT_FATAL(
        packed.dtype() != DataType::UINT8 || args.fp32_dest_acc_en,
        "UINT8 unpack requires fp32_dest_acc_en; use UINT16 for BF16 DEST");
    TT_FATAL(
        args.dtype == DataType::BFLOAT16 || args.dtype == DataType::FLOAT32,
        "NVFP4 output must be BFLOAT16 or FLOAT32");
    TT_FATAL(args.dtype != DataType::FLOAT32 || args.fp32_dest_acc_en, "FLOAT32 output requires fp32_dest_acc_en");
    TT_FATAL(
        args.output_layout == Layout::TILE || args.output_layout == Layout::ROW_MAJOR, "Unsupported output layout");
    TT_FATAL(args.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "NVFP4 output must be interleaved");
    TT_FATAL(
        std::isfinite(args.global_scale) && args.global_scale >= 0,
        "NVFP4 global_scale must be finite and nonnegative");
    TT_FATAL(
        packed.logical_volume() <= std::numeric_limits<uint32_t>::max() / planes,
        "NVFP4 expanded tensor exceeds 32-bit indexing");
}

Nvfp4DeviceOperation::spec_return_value_t Nvfp4DeviceOperation::compute_output_specs(
    const Nvfp4Params& args, const Nvfp4Inputs& inputs) {
    const auto& shape = inputs.packed.logical_shape();
    Shape::Container dims(shape.begin(), shape.end());
    dims.back() *= inputs.packed.element_size() * 2;
    if (args.transpose_output) {
        std::swap(dims[dims.size() - 2], dims.back());
    }
    return {Shape(dims), {args.dtype, {args.output_layout}, args.memory_config}};
}
Nvfp4DeviceOperation::tensor_return_value_t Nvfp4DeviceOperation::create_output_tensors(
    const Nvfp4Params& args, const Nvfp4Inputs& inputs) {
    validate_on_program_cache_miss(args, inputs);
    return create_device_tensor(compute_output_specs(args, inputs), inputs.packed.device());
}
}  // namespace ttnn::experimental::prim
