// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dequantize_mxfp8_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {
using namespace tt::tt_metal;

// Circular-buffer contract shared with the kernels (see README.md). Kernels may
// change how they use these buffers; changing their formats or sizes is a host
// change outside the kernel-only optimization harness.
//   c0  packed E4M3 data, input dtype format (UInt8/UInt16), 2 packed tiles
//   c1  raw E8M0 scale bytes for one packed tile per page, UInt8, 2 pages
//   c2  reader staging for ROW_MAJOR input, UInt8
//   c3  kernel-owned workspace, UInt16 32x32 tiles, 2 * planes pages
//   c4  kernel-owned workspace, Float16_b 32x32 tiles, 2 * planes pages
//   c16 decoded output tiles, output dtype format, 2 * planes pages
//   c17 writer staging for ROW_MAJOR output, UInt8
ProgramDescriptor Mxfp8ProgramFactory::create_descriptor(
    const Mxfp8Params& args, const Mxfp8Inputs& inputs, Tensor& output) {
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t tile_hw = tt::constants::TILE_HW;
    const auto& packed = inputs.packed;
    const uint32_t word_bytes = packed.element_size();
    const uint32_t planes = word_bytes;
    const uint32_t tile_bytes = tile_hw * word_bytes;
    const uint32_t dram_alignment = hal::get_dram_alignment();
    const uint32_t staging_stride = tt::align(tile_width * word_bytes, dram_alignment);
    const uint32_t scale_bytes = planes * tile_height;
    // Blackhole DRAM reads need matching low address bits; keep every scale
    // page at a DRAM-aligned L1 address.
    const uint32_t scale_page_bytes = tt::align(scale_bytes, std::max(dram_alignment, hal::get_l1_alignment()));
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
    add_cb(1, scale_page_bytes, tt::DataFormat::UInt8, 2);
    add_cb(2, (tile_height + 1) * staging_stride, tt::DataFormat::UInt8, 1);
    add_cb(3, tile_hw * 2, tt::DataFormat::UInt16, 2 * planes);
    add_cb(4, tile_hw * 2, tt::DataFormat::Float16_b, 2 * planes);
    add_cb(16, output_tile_bytes, datatype_to_dataformat_converter(output.dtype()), 2 * planes);
    add_cb(17, output_tile_bytes, tt::DataFormat::UInt8, 1);

    const std::string base = "ttnn/cpp/ttnn/operations/experimental/dequantize_mxfp8/device/kernels/";
    std::vector<uint32_t> reader_ctas{
        tile_height,
        tile_width,
        word_bytes,
        width_tiles,
        static_cast<uint32_t>(packed.layout() == Layout::TILE),
        packed.buffer()->page_size(),
        scale_bytes,
        staging_stride};
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
    // Output tiles per packed tile, appended after the tensor accessor.
    writer_ctas.push_back(planes);
    KernelDescriptor writer;
    writer.kernel_source = base + "dataflow/writer.cpp";
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ctas);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = base + "compute/dequantize.cpp";
    compute.core_ranges = all_cores;
    compute.compile_time_args = {
        planes,
        static_cast<uint32_t>(packed.dtype() == DataType::UINT8),
        static_cast<uint32_t>(args.dtype == DataType::BFLOAT16),
        static_cast<uint32_t>(args.fp32_dest_acc_en)};
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
