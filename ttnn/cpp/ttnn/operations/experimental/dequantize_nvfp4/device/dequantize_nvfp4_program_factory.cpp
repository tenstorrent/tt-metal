// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dequantize_nvfp4_program_factory.hpp"

#include <bit>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {
using namespace tt::tt_metal;

ProgramDescriptor Nvfp4ProgramFactory::create_descriptor(
    const Nvfp4Params& args, const Nvfp4Inputs& inputs, Tensor& output) {
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t tile_hw = tt::constants::TILE_HW;
    const auto& packed = inputs.packed;
    const uint32_t word_bytes = packed.element_size();
    const uint32_t planes = word_bytes * 2;
    const uint32_t tile_bytes = tile_hw * word_bytes;
    const uint32_t staging_stride = tt::align(tile_width * word_bytes, hal::get_dram_alignment());
    const uint32_t scale_bytes = planes * 64 * inputs.scales.element_size();
    const uint32_t scale_pages = args.experimental_fast ? planes * 4 : 1;
    const uint32_t output_tile_bytes = tile_hw * output.element_size();
    const uint32_t packed_tiles = packed.logical_volume() / tile_hw;
    const uint32_t width_tiles = packed.logical_shape()[-1] / tile_width;
    const auto grid = packed.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, group1, group2, count1, count2] = split_work_to_cores(grid, packed_tiles);
    ProgramDescriptor desc;
    auto add_cb = [&](uint8_t id, uint32_t page_size, tt::DataFormat format, uint32_t pages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size * pages,
            .core_ranges = all_cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = id, .data_format = format, .page_size = page_size}}}});
    };
    add_cb(0, tile_bytes, datatype_to_dataformat_converter(packed.dtype()), 2);
    // Precise-mode scales stay raw; fast mode unpacks a compact BF16 scale tile.
    if (args.experimental_fast) {
        // Each page holds 16 scales. Sixteen contiguous pages form one 16x16
        // tile, copied to DEST once and expanded into per-face scale vectors.
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * scale_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = 1,
                .data_format = tt::DataFormat::Float16_b,
                .page_size = 32,
                .tile = TileDescriptor(Tile({16, 16}))}}}});
        if (args.fpu_tensor_scale) {
            add_cb(3, output_tile_bytes, tt::DataFormat::Float16_b, 1);
        }
    } else {
        add_cb(1, scale_bytes, tt::DataFormat::UInt8, 2);
    }
    add_cb(2, (tile_height + 1) * staging_stride, tt::DataFormat::UInt8, 1);
    add_cb(16, output_tile_bytes, datatype_to_dataformat_converter(output.dtype()), 2 * planes);
    add_cb(17, output_tile_bytes, tt::DataFormat::UInt8, 1);

    const std::string base = "ttnn/cpp/ttnn/operations/experimental/dequantize_nvfp4/device/kernels/";
    std::vector<uint32_t> reader_ctas{
        tile_height,
        tile_width,
        word_bytes,
        width_tiles,
        static_cast<uint32_t>(packed.layout() == Layout::TILE),
        packed.buffer()->page_size(),
        scale_bytes,
        staging_stride,
        scale_pages,
        static_cast<uint32_t>(args.experimental_fast),
        std::bit_cast<uint32_t>(args.global_scale),
        static_cast<uint32_t>(args.fpu_tensor_scale)};
    TensorAccessorArgs(*packed.buffer()).append_to(reader_ctas);
    TensorAccessorArgs(*inputs.scales.buffer()).append_to(reader_ctas);
    KernelDescriptor reader;
    reader.kernel_source = base + "dataflow/reader.cpp";
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ctas);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ctas{
        tile_height,
        tile_width,
        output.element_size(),
        width_tiles * planes,
        static_cast<uint32_t>(output.layout() == Layout::TILE),
        output.buffer()->page_size()};
    TensorAccessorArgs(*output.buffer()).append_to(writer_ctas);
    // Keep the tensor accessor at offset 6; append transpose metadata after it.
    writer_ctas.push_back(static_cast<uint32_t>(args.transpose_output));
    writer_ctas.push_back(packed.logical_shape()[-2] / tile_height);
    writer_ctas.push_back(width_tiles * planes);
    KernelDescriptor writer;
    writer.kernel_source = base + "dataflow/writer.cpp";
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ctas);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source =
        base + "compute/" + std::string(args.experimental_fast ? "dequantize_fast.cpp" : "dequantize.cpp");
    compute.core_ranges = all_cores;
    compute.compile_time_args = {
        planes,
        static_cast<uint32_t>(packed.dtype() == DataType::UINT8),
        static_cast<uint32_t>(args.scale_format == operations::experimental::Nvfp4ScaleFormat::E2M1),
        std::bit_cast<uint32_t>(args.global_scale),
        static_cast<uint32_t>(args.dtype == DataType::BFLOAT16),
        static_cast<uint32_t>(args.fpu_tensor_scale),
        static_cast<uint32_t>(args.transpose_output)};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = args.fp32_dest_acc_en, .dst_full_sync_en = true};

    uint32_t start = 0;
    for (const auto& core : grid_to_cores(num_cores, grid.x, grid.y)) {
        const uint32_t count = group1.contains(core) ? count1 : count2;
        reader.emplace_runtime_args(core, {packed.buffer(), inputs.scales.buffer(), start, count});
        writer.emplace_runtime_args(core, {output.buffer(), start * planes, count * planes});
        compute.emplace_runtime_args(core, {count});
        start += count;
    }
    desc.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    return desc;
}
}  // namespace ttnn::experimental::prim
