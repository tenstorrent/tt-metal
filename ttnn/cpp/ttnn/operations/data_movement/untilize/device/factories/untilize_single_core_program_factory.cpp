// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "untilize_single_core_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {
tt::tt_metal::ProgramDescriptor UntilizeSingleCoreProgramFactory::create_descriptor(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    ProgramDescriptor desc;

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_volume = tile_height * tile_width;

    uint32_t num_tiles = a.physical_volume() / tile_volume;
    uint32_t num_blocks_across_height = a.physical_volume() / a.padded_shape()[-1] / tile_height;
    uint32_t num_columns_of_blocks = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        uint32_t output_shard_width;
        if (output.shard_spec().has_value()) {
            output_shard_width = output.shard_spec().value().shape[1];
        } else {
            output_shard_width = output.nd_shard_spec().value().shard_shape[-1];
        }
        num_columns_of_blocks = a.padded_shape()[-1] / output_shard_width;
    }
    uint32_t num_tiles_per_column_row = a.padded_shape()[-1] / num_columns_of_blocks / tile_width;

    // Determine how much L1 space we can use for input and output CBs,
    // ensuring that we don't intrude into other L1 storage space
    uint32_t max_l1_size =
        (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Determine the max number of tiles that can be in any CB at a given time (1 input CB + 1 output CB = 2 total CBs)
    uint32_t max_tiles_per_cb = max_l1_size / (input_single_tile_size + output_single_tile_size);

    // Determine how many tiles each block will store.
    // Currently we require that the number of tiles in a row is divisible by the number of blocks in a row, or
    // equivalently the number of tiles in a row is divisible by the number of tiles in a block.
    uint32_t num_tiles_per_block = num_tiles_per_column_row;
    if (num_tiles_per_block > max_tiles_per_cb) {
        for (uint32_t i = max_tiles_per_cb; i > 0; --i) {
            if (num_tiles_per_column_row % i == 0) {
                num_tiles_per_block = i;
                break;
            }
        }
    }

    uint32_t num_blocks_per_column_row = num_tiles_per_column_row / num_tiles_per_block;
    uint32_t output_single_block_width_size = num_tiles_per_block * TILE_WIDTH * output.element_size();
    uint32_t num_total_sticks = a.physical_volume() / a.padded_shape()[-1] * num_columns_of_blocks;
    uint32_t output_stick_size = a.physical_volume() * output.element_size() / num_total_sticks;

    // Input CB
    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t input_cb_num_tiles = num_tiles_per_block;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_num_tiles * input_single_tile_size,
        .core_ranges = core,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    // Output CB
    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;
    uint32_t output_cb_num_tiles = num_tiles_per_block;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_num_tiles * output_single_tile_size,
        .core_ranges = core,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // Reader compile-time args
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};

    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    // Tilized reader
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "reader_unary_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer compile-time args
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)output_cb_index,
        (uint32_t)output_stick_size,
        (uint32_t)tile_height,
        (uint32_t)num_blocks_across_height,
        (uint32_t)num_columns_of_blocks,
        (uint32_t)num_blocks_per_column_row,
        (uint32_t)num_tiles_per_block,
        (uint32_t)output_single_block_width_size,
    };
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Untilized writer
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_single_core.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Compute defines
    KernelDescriptor::Defines compute_kernel_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    // Compute compile-time args
    uint32_t num_blocks = num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height;
    std::vector<uint32_t> compute_compile_time_args = {
        (uint32_t)num_blocks, (uint32_t)num_tiles_per_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core;
    compute_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_desc.defines = std::move(compute_kernel_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    // Reader run-time args
    uint32_t start_page_id = 0;
    reader_desc.emplace_runtime_args(CoreCoord{0, 0}, {src0_buffer, num_tiles, start_page_id});

    // Writer run-time args
    writer_desc.emplace_runtime_args(CoreCoord{0, 0}, {dst_buffer});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}
}  // namespace ttnn::prim
