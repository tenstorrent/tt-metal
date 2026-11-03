// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dequantize_mxfp8_device_operation.hpp"

#include "ttnn/tensor/tensor_ops.hpp"

#include <limits>
#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::experimental::prim {
void Mxfp8DeviceOperation::validate_on_program_cache_miss(const Mxfp8Params& args, const Mxfp8Inputs& inputs) {
    using namespace tt::tt_metal;
    for (const auto* tensor : {&inputs.packed, &inputs.scales}) {
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "MXFP8 inputs must be device tensors");
        TT_FATAL(tensor->buffer() != nullptr, "MXFP8 input buffer is null");
        TT_FATAL(
            !tensor->is_sharded() && tensor->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "MXFP8 requires interleaved inputs");
        TT_FATAL(tensor->logical_shape() == tensor->padded_shape(), "MXFP8 inputs must have no padding");
    }
    const auto& packed = inputs.packed;
    TT_FATAL(packed.device() == inputs.scales.device(), "MXFP8 inputs must be on the same device");
    TT_FATAL(packed.device()->arch() == tt::ARCH::BLACKHOLE, "MXFP8 dequantization requires Blackhole");
    TT_FATAL(
        packed.dtype() == DataType::UINT8 || packed.dtype() == DataType::UINT16,
        "MXFP8 packed data must be UINT8 or UINT16");
    TT_FATAL(packed.layout() == Layout::ROW_MAJOR || packed.layout() == Layout::TILE, "Unsupported packed layout");
    const auto& shape = packed.logical_shape();
    TT_FATAL(shape.size() >= 2 && shape.size() <= 4, "MXFP8 packed data must have rank 2 to 4");
    TT_FATAL(
        shape[-2] > 0 && shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] > 0 &&
            shape[-1] % tt::constants::TILE_WIDTH == 0,
        "MXFP8 packed height and width must be positive multiples of 32");
    TT_FATAL(packed.tensor_spec().tile() == Tile({32, 32}), "MXFP8 requires standard 32x32 tiles");
    const uint32_t planes = packed.element_size();
    const uint32_t packed_tiles = packed.logical_volume() / tt::constants::TILE_HW;
    TT_FATAL(
        inputs.scales.dtype() == DataType::UINT8 && inputs.scales.layout() == Layout::ROW_MAJOR,
        "MXFP8 scales must be row-major UINT8 E8M0 codes");
    TT_FATAL(
        inputs.scales.logical_shape() == Shape({packed_tiles, planes * tt::constants::TILE_HEIGHT}),
        "MXFP8 scales must have shape [packed_tile_count, 32 * values_per_word]");
    TT_FATAL(
        args.dtype == DataType::BFLOAT16 || args.dtype == DataType::FLOAT32,
        "MXFP8 output must be BFLOAT16 or FLOAT32");
    TT_FATAL(args.dtype != DataType::FLOAT32 || args.fp32_dest_acc_en, "FLOAT32 output requires fp32_dest_acc_en");
    TT_FATAL(
        args.output_layout == Layout::TILE || args.output_layout == Layout::ROW_MAJOR, "Unsupported output layout");
    TT_FATAL(args.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "MXFP8 output must be interleaved");
    TT_FATAL(
        packed.logical_volume() <= std::numeric_limits<uint32_t>::max() / planes,
        "MXFP8 expanded tensor exceeds 32-bit indexing");
}

Mxfp8DeviceOperation::spec_return_value_t Mxfp8DeviceOperation::compute_output_specs(
    const Mxfp8Params& args, const Mxfp8Inputs& inputs) {
    const auto& shape = inputs.packed.logical_shape();
    Shape::Container dims(shape.begin(), shape.end());
    dims.back() *= inputs.packed.element_size();
    return {Shape(dims), {args.dtype, {args.output_layout}, args.memory_config}};
}

Mxfp8DeviceOperation::tensor_return_value_t Mxfp8DeviceOperation::create_output_tensors(
    const Mxfp8Params& args, const Mxfp8Inputs& inputs) {
    validate_on_program_cache_miss(args, inputs);
    return create_device_tensor(compute_output_specs(args, inputs), inputs.packed.device());
}
}  // namespace ttnn::experimental::prim
