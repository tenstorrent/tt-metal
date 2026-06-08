// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_single_core_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor TilizeSingleCoreProgramFactory::create_descriptor(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreRangeSet core_ranges{core};

    Buffer* src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device

    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;

    uint32_t num_tiles = a.physical_volume() / TILE_HW;

    auto width = a.padded_shape()[-1];
    uint32_t stick_s = width;
    uint32_t num_sticks = a.physical_volume() / width;
    uint32_t stick_size = stick_s * a.element_size();  // Assuming bfloat16 dataformat

    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    uint32_t num_tiles_per_block = 1;

    if (!operation_attributes.use_low_perf) {
        // Ensure we don't intrude into storage space
        uint32_t max_l1_size =
            (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size);  // 2 CBs
        // Currently need the number of tiles in a row to be divisible by tiles in a block
        if (num_tiles_in_row <= max_tiles) {
            num_tiles_per_block = num_tiles_in_row;
        } else {
            for (uint32_t n_t = max_tiles; n_t > 0; n_t--) {
                if (num_tiles_in_row % n_t == 0) {
                    num_tiles_per_block = n_t;
                    break;
                }
            }
        }
    }

    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * a.element_size();

    const uint32_t src0_cb_index = 0;
    const uint32_t output_cb_index = tt::CBIndex::c_16;
    const uint32_t num_input_tiles = num_tiles_per_block;
    const uint32_t num_output_tiles = num_tiles_per_block;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // Reader compile-time args
    std::vector<uint32_t> reader_compile_time_args = {stick_size};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Tilized reader
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
        "reader_unary_stick_layout_split_rows_singlecore.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_ranges;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Buffer* slots register BufferBindings so the framework patches addresses on cache hits.
    reader_desc.emplace_runtime_args(
        core.start_coord,
        {src0_buffer,
         num_sticks,
         stick_size,
         num_tiles_per_block,
         block_width_size,
         num_full_blocks_in_row,
         num_leftover_tiles,
         leftover_width_in_row,
         std::uint32_t{0}});  // row_start_id

    // Tilized writer
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_ranges;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.emplace_runtime_args(core.start_coord, {dst_buffer, num_tiles, std::uint32_t{0}});

    std::vector<uint32_t> compute_args = {
        num_tiles / num_tiles_per_block,  // per_core_block_cnt
        num_tiles_per_block               // per_core_block_tile_cnt
    };

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_llk_acc) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/kernel/compute/tilize.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_ranges;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_llk_acc,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
