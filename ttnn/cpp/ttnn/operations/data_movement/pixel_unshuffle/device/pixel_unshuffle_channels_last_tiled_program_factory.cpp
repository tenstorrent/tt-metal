// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_unshuffle, channels_last, COLUMN_MAJOR_PADDED: the sub-pixel gather done by the compute engine.
//
// For one output image row `oh`, the C*r input rows (c, dy) = rows oh*r+dy of every channel form a
// [C*r, W] row-major block whose columns are (ow, dx) with dx fastest. Tilizing that block (pad rows
// C*r..31 are zero) gives, per 32 input columns, a tile T[(c,dy), (ow_local, dx)]; transposing it on the
// FPU gives T'[(ow_local, dx), (c,dy)]; untilizing T' alone writes its 32 rows as 32 contiguous 64 B
// rows = (32/r) output pixels, each a stick of r x 32 channels: channel rw*32 + rh*C + c, zero-padded
// per sub-pixel column. So every pixel's stick lands in the output shard with no RISC data movement at
// all: the NCRISC only reads C*r whole input rows per item from DRAM, the compute kernel does
// tilize -> transpose_tile -> pack_untilize on batches of up to 8 tiles.
//
// Geometry: 2-byte elements, r | 32, C*r <= 32, W % 32 == 0, shard height a multiple of 32/r pixels.
#include "pixel_unshuffle_device_op.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/common/constants.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor PixelUnshuffle::MultiCoreChannelsLast::create_descriptor_tiled(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = output_tensor;
    const auto& in_shape = input.logical_shape();
    const uint32_t N = in_shape[0];
    const uint32_t C = in_shape[1];
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];
    const uint32_t r = op_attr.downscale_factor;
    const uint32_t Ho = H / r;
    const uint32_t Wo = W / r;
    const uint32_t Cp = op_attr.padded_channels;  // r * 32

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(src_buffer != nullptr && dst_buffer != nullptr, "PixelUnshuffle: buffers must be allocated.");

    const tt::DataFormat in_data_fmt = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat out_data_fmt = datatype_to_dataformat_converter(output.dtype());
    const uint32_t datum_sz = tt::datum_size(in_data_fmt);
    TT_FATAL(datum_sz == 2, "PixelUnshuffle(COLUMN_MAJOR_PADDED): 2-byte elements only.");
    const uint32_t row_bytes = W * datum_sz;      // one input image row (W % 32 == 0: 64 B aligned)
    const uint32_t Wt = W / TILE_WIDTH;           // tiles per image row
    const uint32_t ppt = TILE_WIDTH / r;          // output pixels per tile
    constexpr uint32_t tile_batch = 8;            // tiles per DEST acquisition (half-sync, 16-bit)
    const uint32_t chunk_tiles = std::min<uint32_t>(Wt, 64);  // tiles tilized per unpacker/packer reconfiguration (cb_til capacity, <= 128 KB)
    const uint32_t tile_bytes = TILE_HW * datum_sz;
    const uint32_t stick_bytes = static_cast<uint32_t>(dst_buffer->aligned_page_size());
    TT_FATAL(
        stick_bytes == Cp * tt::datum_size(out_data_fmt) && Cp == r * TILE_WIDTH,
        "PixelUnshuffle(COLUMN_MAJOR_PADDED): output stick {} B must be r*32 channels x {} B.",
        stick_bytes,
        tt::datum_size(out_data_fmt));
    TT_FATAL(ppt * stick_bytes == TILE_HEIGHT * TILE_WIDTH * datum_sz, "PixelUnshuffle: a tile must untilize into exactly {} sticks.", ppt);

    const auto& shard_spec = output.shard_spec().value();
    const uint32_t shard_h = shard_spec.shape[0];
    const bool row_wise = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const std::vector<CoreCoord> cores = corerange_to_cores(shard_spec.grid, std::nullopt, row_wise);
    const CoreRangeSet& all_cores = shard_spec.grid;
    const uint32_t total_pixels = N * Ho * Wo;

    ProgramDescriptor desc;
    constexpr uint32_t cb_in = tt::CBIndex::c_0;    // [32 rows x W] row-major input block, double-buffered
    constexpr uint32_t cb_til = tt::CBIndex::c_24;  // tilized tiles of one chunk
    constexpr uint32_t cb_out = tt::CBIndex::c_16;  // the output shard: sticks
    const uint32_t in_page_bytes = TILE_HEIGHT * row_bytes;
    {
        CBDescriptor cb;
        cb.total_size = 2 * in_page_bytes;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in), .data_format = in_data_fmt, .page_size = in_page_bytes});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = chunk_tiles * tile_bytes;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_til), .data_format = in_data_fmt, .page_size = tile_bytes});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(dst_buffer->aligned_size_per_bank());
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_out), .data_format = out_data_fmt, .page_size = stick_bytes});
        cb.buffer = dst_buffer;
        desc.cbs.push_back(std::move(cb));
    }

    KernelDescriptor reader;
    {
        std::vector<uint32_t> cta = {W, C, H, r, Wt, cb_in, row_bytes};
        TensorAccessorArgs(*src_buffer).append_to(cta);
        reader.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/dataflow/"
            "pixel_unshuffle_nhwc_tiled_reader.cpp";
        reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader.core_ranges = all_cores;
        reader.compile_time_args = std::move(cta);
        reader.config = ReaderConfigDescriptor{};
    }
    KernelDescriptor compute;
    {
        compute.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/compute/"
            "pixel_unshuffle_nhwc_tiled.cpp";
        compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute.core_ranges = all_cores;
        compute.compile_time_args = {Wt, C, r, ppt, tile_batch, cb_in, cb_til, cb_out, chunk_tiles};
        compute.config = ComputeConfigDescriptor{};
    }
    for (uint32_t i = 0; i < cores.size(); i++) {
        const uint32_t pix0 = std::min(i * shard_h, total_pixels);
        const uint32_t npix = std::min(shard_h, total_pixels - pix0);
        reader.emplace_runtime_args(cores[i], {src_buffer, pix0 / ppt, npix / ppt});
        compute.emplace_runtime_args(cores[i], {pix0 / ppt, npix / ppt});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::operations::data_movement
