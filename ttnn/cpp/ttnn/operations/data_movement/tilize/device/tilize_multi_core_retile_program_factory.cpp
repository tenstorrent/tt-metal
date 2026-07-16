// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_retile_program_factory.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor TilizeMultiCoreRetileProgramFactory::create_descriptor(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;

    // Input is already tiled; its tile shape differs from the requested output tile shape.
    const Tile& input_tile = a.tensor_spec().tile();
    const Tile& output_tile = operation_attributes.tile;

    const uint32_t in_tile_width = input_tile.get_width();
    const uint32_t in_tile_height = input_tile.get_height();
    const uint32_t out_tile_width = output_tile.get_width();
    const uint32_t out_tile_height = output_tile.get_height();

    TT_FATAL(
        in_tile_width == TILE_WIDTH && out_tile_width == TILE_WIDTH,
        "Retile requires tile width {}, got input {} and output {}",
        TILE_WIDTH,
        in_tile_width,
        out_tile_width);
    TT_FATAL(
        in_tile_height >= out_tile_height && (in_tile_height % out_tile_height) == 0,
        "Retile currently supports in_tile_height >= out_tile_height with exact divisibility; got {} -> {}",
        in_tile_height,
        out_tile_height);
    TT_FATAL(
        !a.is_sharded() && output.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Retile program factory currently supports interleaved input/output only");

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t input_single_tile_size = input_tile.get_tile_size(input_cb_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_cb_data_format);
    const uint32_t mid_page_size = std::max(input_single_tile_size, output_single_tile_size);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src0_buffer != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto& padded_shape = a.padded_shape();
    const uint32_t tensor_width = padded_shape[-1];
    const uint32_t tensor_height = a.physical_volume() / tensor_width;

    TT_FATAL(tensor_width % in_tile_width == 0, "Tensor width must be divisible by input tile width");
    TT_FATAL(tensor_height % in_tile_height == 0, "Tensor height must be divisible by input tile height");
    TT_FATAL(tensor_height % out_tile_height == 0, "Tensor height must be divisible by output tile height");

    const uint32_t tiles_per_block = tensor_width / in_tile_width;  // == width / out_tile_width
    const uint32_t num_input_tile_rows = tensor_height / in_tile_height;
    const uint32_t height_ratio = in_tile_height / out_tile_height;

    auto* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, num_input_tile_rows);

    // Double-buffer when a core processes more than one block so reader/compute/writer overlap.
    const uint32_t cb_num_pages_per_block = tiles_per_block;
    const uint32_t cb_factor = (nblocks_per_core > 1 || nblocks_per_core_cliff > 1) ? 2 : 1;
    const uint32_t src_cb_tiles = cb_num_pages_per_block * cb_factor;
    const uint32_t mid_cb_pages = cb_num_pages_per_block * cb_factor;
    const uint32_t out_cb_tiles = cb_num_pages_per_block * cb_factor;

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t mid_untilize_cb_index = tt::CBIndex::c_1;
    constexpr uint32_t mid_tilize_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;

    ProgramDescriptor desc;

    // Input CB (tiled, input tile shape) — double-buffered for reader/compute overlap.
    desc.cbs.push_back(CBDescriptor{
        .total_size = src_cb_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = input_tile,
        }}},
    });

    // Intermediate RM CB for untilize output. Page size = max(in, out) tile size.
    // Tile/face geometry matches the input tile (required by pack_untilize).
    // Kernels must not call get_tile_size() on this CB — they use mid_page_size CT arg.
    desc.cbs.push_back(CBDescriptor{
        .total_size = mid_cb_pages * mid_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(mid_untilize_cb_index),
            .data_format = input_cb_data_format,
            .page_size = mid_page_size,
            .tile = input_tile,
        }}},
    });

    // Intermediate RM CB for tilize input. Same page size; output tile face geometry.
    desc.cbs.push_back(CBDescriptor{
        .total_size = mid_cb_pages * mid_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(mid_tilize_cb_index),
            .data_format = output_cb_data_format,
            .page_size = mid_page_size,
            .tile = output_tile,
        }}},
    });

    // Output CB (tiled, output tile shape) — double-buffered for compute/writer overlap.
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_cb_tiles * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile,
        }}},
    });

    // Reader: interleaved tiled pages.
    {
        std::vector<uint32_t> reader_ct_args = {src0_cb_index};
        TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
            "reader_unary_start_id.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.config = ReaderConfigDescriptor{};
        desc.kernels.push_back(std::move(reader_desc));
    }

    // Writer: interleaved tiled pages.
    {
        std::vector<uint32_t> writer_ct_args = {output_cb_index};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};
        desc.kernels.push_back(std::move(writer_desc));
    }

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_llk_acc) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[mid_tilize_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    auto make_compute_desc = [&](const CoreRangeSet& ranges) {
        KernelDescriptor cd;
        cd.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/retile.cpp";
        cd.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd.core_ranges = ranges;
        cd.compile_time_args = {
            tiles_per_block,
            src0_cb_index,
            mid_untilize_cb_index,
            mid_tilize_cb_index,
            output_cb_index,
            in_tile_height,
            out_tile_height,
            mid_page_size,
            output_single_tile_size,
        };
        cd.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_llk_acc,
            .unpack_to_dest_mode = unpack_to_dest_mode,
        };
        return cd;
    };

    constexpr size_t reader_idx = 0;
    constexpr size_t writer_idx = 1;
    int full_compute_idx = -1;
    int cliff_compute_idx = -1;

    if (!core_range.ranges().empty()) {
        full_compute_idx = static_cast<int>(desc.kernels.size());
        desc.kernels.push_back(make_compute_desc(core_range));
    }
    if (!core_range_cliff.empty()) {
        cliff_compute_idx = static_cast<int>(desc.kernels.size());
        desc.kernels.push_back(make_compute_desc(core_range_cliff));
    }

    KernelDescriptor& reader_ref = desc.kernels[reader_idx];
    KernelDescriptor& writer_ref = desc.kernels[writer_idx];

    const bool has_cliff = !core_range_cliff.empty();
    const uint32_t ncores_full = ncores - (has_cliff ? 1 : 0);
    uint32_t input_tile_start_id = 0;
    uint32_t output_tile_start_id = 0;
    const auto& cores = corerange_to_cores(all_cores);

    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t num_input_blocks = nblocks_per_core;
        const uint32_t num_input_tiles = num_input_blocks * tiles_per_block;
        const uint32_t num_output_tiles = num_input_blocks * height_ratio * tiles_per_block;

        reader_ref.emplace_runtime_args(core, {src0_buffer, num_input_tiles, input_tile_start_id});
        writer_ref.emplace_runtime_args(core, {dst_buffer, num_output_tiles, output_tile_start_id});
        if (full_compute_idx >= 0) {
            desc.kernels[full_compute_idx].emplace_runtime_args(core, {num_input_blocks});
        }

        input_tile_start_id += num_input_tiles;
        output_tile_start_id += num_output_tiles;
    }

    if (has_cliff) {
        const CoreCoord& core = cores[ncores_full];
        const uint32_t num_input_blocks = nblocks_per_core_cliff;
        const uint32_t num_input_tiles = num_input_blocks * tiles_per_block;
        const uint32_t num_output_tiles = num_input_blocks * height_ratio * tiles_per_block;

        reader_ref.emplace_runtime_args(core, {src0_buffer, num_input_tiles, input_tile_start_id});
        writer_ref.emplace_runtime_args(core, {dst_buffer, num_output_tiles, output_tile_start_id});
        if (cliff_compute_idx >= 0) {
            desc.kernels[cliff_compute_idx].emplace_runtime_args(core, {num_input_blocks});
        }
    }

    return desc;
}

}  // namespace ttnn::prim
