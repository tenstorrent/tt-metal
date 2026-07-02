// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include "untilize_multi_core_program_factory.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor UntilizeMultiCoreProgramFactory::create_descriptor(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& output) {
    ProgramDescriptor desc;

    const auto& a = tensor_args.input;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    tt::tt_metal::IDevice* device = a.device();
    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t tensor_width = a.padded_shape()[-1];
    uint32_t tensor_height = a.physical_volume() / tensor_width;

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    bool input_is_sharded = a.is_sharded();
    std::vector<CoreCoord> ordered_cores_with_data;

    uint32_t num_tiles_per_row = tensor_width / tile_width;
    uint32_t num_tiles_per_col = tensor_height / tile_height;

    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_compute_cores,
         compute_core_range,
         full_compute_core_range,
         cliff_compute_core_range,
         num_rows_per_full_core,
         num_rows_per_cliff_core] = ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);

    // Default values are for interleaved input.
    // Cliff core applicable interleaved input only, it is the only core not processing the
    // same number of rows (blocks) as all other cores.
    uint32_t num_input_blocks_across_width = 1;
    uint32_t num_tiles_per_input_block = num_tiles_per_row;
    uint32_t num_input_blocks_per_full_core = num_rows_per_full_core;
    uint32_t num_input_blocks_per_cliff_core = num_rows_per_cliff_core;
    uint32_t input_shard_height = 0;
    uint32_t input_shard_width = 0;
    if (input_is_sharded) {
        ShardSpec input_shard_spec = a.shard_spec().value();
        input_shard_height = input_shard_spec.shape[0];
        input_shard_width = input_shard_spec.shape[1];
        num_compute_cores = input_shard_spec.grid.num_cores();

        // Note: Accounting for uneven input shards
        num_input_blocks_across_width = tt::div_up(tensor_width, input_shard_width);
        num_tiles_per_input_block = input_shard_width / tile_width;
        num_input_blocks_per_full_core = input_shard_height / tile_height;
        num_input_blocks_per_cliff_core = 0;

        ordered_cores_with_data = get_optimal_worker_cores_for_sharded_tensor(a);
        compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));
        full_compute_core_range = compute_core_range;
        cliff_compute_core_range = CoreRangeSet();
    }

    // Detect if sharding is uneven - works for HEIGHT, WIDTH, and BLOCK sharding
    bool has_uneven_sharding = false;
    if (input_is_sharded) {
        uint32_t height_remainder = tensor_height % input_shard_height;
        uint32_t width_remainder = tensor_width % input_shard_width;
        has_uneven_sharding = (height_remainder != 0) || (width_remainder != 0);
    }

    const bool input_is_dram_sharded = input_is_sharded && src0_buffer->buffer_type() == BufferType::DRAM;

    // Block reader: unbacked double-buffer CB, reads from L1 shard block-by-block.
    // Required for uneven sharding where CB backing has a size mismatch.
    // Even sharding uses zero-copy backed CB (fast production path).
    bool use_block_reader = input_is_sharded && (has_uneven_sharding || input_is_dram_sharded);

    // Input CB
    uint32_t input_cb_num_tiles;
    if (input_is_sharded && !use_block_reader) {
        // Even sharding with pack_untilize: CB is backed by the sharded buffer (zero-copy)
        input_cb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core;
    } else {
        // Block reader (sharded) or interleaved: double-buffer
        input_cb_num_tiles =
            (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    }
    Buffer* cb_backing_buffer = (input_is_sharded && !use_block_reader) ? src0_buffer : nullptr;
    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_num_tiles * input_single_tile_size,
        .core_ranges = compute_core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
        .buffer = cb_backing_buffer,
    });

    // Output CB
    uint32_t output_cb_num_tiles;
    if (num_input_blocks_per_full_core == 1) {
        // No need to double buffer if the core is only processing a single block
        output_cb_num_tiles = num_tiles_per_input_block;
    } else {
        // Double buffer if the core is processing 2+ blocks
        output_cb_num_tiles = num_tiles_per_input_block * 2;
    }
    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_num_tiles * output_single_tile_size,
        .core_ranges = compute_core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // Reader compile-time args and kernel
    KernelDescriptor reader_desc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = compute_core_range;
    if (use_block_reader) {
        // Block reader: copies from L1 shard into double-buffered CB one block at a time
        // or reads from DRAM shards via TensorAccessor.
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
            "reader_unary_sharded_blocks.cpp";
        reader_desc.compile_time_args = {
            (uint32_t)src0_cb_index,
            (uint32_t)num_tiles_per_input_block,
        };
        TensorAccessorArgs(*src0_buffer).append_to(reader_desc.compile_time_args);
        reader_desc.config = ReaderConfigDescriptor{};
    } else if (input_is_sharded) {
        // Even sharding with pack_untilize: CB is backed by the sharded buffer, reader just pushes
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
        reader_desc.compile_time_args = {(uint32_t)src0_cb_index};
        reader_desc.config = ReaderConfigDescriptor{};
    } else {
        // Interleaved input
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
            "reader_unary_start_id.cpp";
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
        reader_desc.compile_time_args = std::move(reader_compile_time_args);
        reader_desc.config = ReaderConfigDescriptor{};
    }

    // Writer compile-time args
    uint32_t output_element_size = output.element_size();
    uint32_t output_page_width =
        tensor_width;  // In height-sharded and interleaved cases, the output page is the entire tensor row
    uint32_t output_num_blocks_across_width = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        if (output.shard_spec().has_value()) {
            output_page_width = output.shard_spec().value().shape[1];
        } else {
            output_page_width = output.nd_shard_spec().value().shard_shape[-1];
        }
        output_num_blocks_across_width = tt::div_up(tensor_width, output_page_width);
    }

    uint32_t num_cols_per_input_block = num_tiles_per_input_block * tile_width;
    uint32_t num_cols_per_output_block = output_page_width;
    uint32_t output_stick_size = num_cols_per_output_block * output_element_size;
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)output_cb_index,
        (uint32_t)output_stick_size,
        (uint32_t)tile_height,
        (uint32_t)num_tiles_per_input_block,
        (uint32_t)output_num_blocks_across_width,
        (uint32_t)output_element_size,
        (uint32_t)num_cols_per_input_block,
        (uint32_t)num_cols_per_output_block,
    };
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multi_core.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = compute_core_range;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    // Compute kernel file
    const std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp");

    KernelDescriptor::Defines compute_kernel_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }

    // Push reader and writer first; track indices for the compute kernel(s) which may be
    // absent depending on whether the corresponding core range is empty (sharded input never
    // has a cliff core, for example).
    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    constexpr size_t reader_idx = 0;
    constexpr size_t writer_idx = 1;
    int full_compute_idx = -1;
    int cliff_compute_idx = -1;

    // Compute compile-time args and kernel
    // Note: This condition is always true for sharded input
    if (!full_compute_core_range.ranges().empty()) {
        std::vector<uint32_t> compute_compile_time_args = {
            (uint32_t)num_tiles_per_input_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};
        KernelDescriptor compute_desc;
        compute_desc.kernel_source = compute_kernel;
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = full_compute_core_range;
        compute_desc.compile_time_args = std::move(compute_compile_time_args);
        compute_desc.defines = compute_kernel_defines;
        compute_desc.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
        };
        full_compute_idx = static_cast<int>(desc.kernels.size());
        desc.kernels.push_back(std::move(compute_desc));
    }

    // Compute Cliff compile_time args and kernel
    // Note: This condition is always false for sharded input (sharded input will never have a cliff core)
    if (!cliff_compute_core_range.ranges().empty()) {
        std::vector<uint32_t> compute_compile_time_args_cliff = {
            (uint32_t)num_tiles_per_input_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};
        KernelDescriptor cliff_desc;
        cliff_desc.kernel_source = compute_kernel;
        cliff_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cliff_desc.core_ranges = cliff_compute_core_range;
        cliff_desc.compile_time_args = std::move(compute_compile_time_args_cliff);
        cliff_desc.defines = std::move(compute_kernel_defines);
        cliff_desc.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        };
        cliff_compute_idx = static_cast<int>(desc.kernels.size());
        desc.kernels.push_back(std::move(cliff_desc));
    }

    KernelDescriptor& reader_ref = desc.kernels[reader_idx];
    KernelDescriptor& writer_ref = desc.kernels[writer_idx];

    // Run-time arg assignment
    // Note: This variable is only applicable to interleaved input
    uint32_t tile_start_index = 0;

    // Run-time args (full cores)
    // Note: For sharded input, these are the only cores used
    bool is_row_major = input_is_sharded ? a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR : true;
    std::vector<CoreCoord> full_cores = input_is_sharded
                                            ? ordered_cores_with_data
                                            : corerange_to_cores(full_compute_core_range, std::nullopt, is_row_major);
    for (uint32_t i = 0; i < full_cores.size(); ++i) {
        CoreCoord core = full_cores[i];
        uint32_t height_wise_input_block_start_index =
            (i / num_input_blocks_across_width) * num_input_blocks_per_full_core;
        uint32_t width_wise_input_block_index = i % num_input_blocks_across_width;

        // Handle uneven input sharding width wise (writer run-time arg)
        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;
        if (input_is_sharded) {
            bool is_last_input_shard_in_row = width_wise_input_block_index == num_input_blocks_across_width - 1;
            if (is_last_input_shard_in_row) {
                uint32_t input_shard_width = a.shard_spec().value().shape[1];
                num_unpadded_cols_per_input_block =
                    num_cols_per_input_block - (tt::round_up(tensor_width, input_shard_width) - tensor_width);
            }
        }

        // Handle uneven input sharding height wise (reader, compute, writer run-time arg)
        uint32_t num_input_blocks_to_process = num_input_blocks_per_full_core;
        if (input_is_sharded) {
            uint32_t input_shard_height = a.shard_spec().value().shape[0];
            uint32_t height_wise_shard_index = i / num_input_blocks_across_width;
            uint32_t num_shards_height_wise = tt::div_up(tensor_height, input_shard_height);
            bool is_last_input_shard_in_col = height_wise_shard_index == num_shards_height_wise - 1;
            if (is_last_input_shard_in_col) {
                num_input_blocks_to_process =
                    num_input_blocks_per_full_core -
                    (tt::round_up(tensor_height, input_shard_height) - tensor_height) / tile_height;
            }
        }

        // Reader run-time args
        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        if (use_block_reader) {
            reader_ref.emplace_runtime_args(
                core,
                {
                    src0_buffer,
                    i,
                    num_input_blocks_to_process,
                });
        } else if (input_is_sharded) {
            reader_ref.emplace_runtime_args(core, {num_tiles_to_read});
        } else {
            // Interleaved input
            reader_ref.emplace_runtime_args(core, {src0_buffer, num_tiles_to_read, tile_start_index});
        }

        // Writer run-time args
        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        writer_ref.emplace_runtime_args(
            core,
            {dst_buffer,
             num_input_blocks_to_process,
             height_wise_input_block_start_index,
             num_unpadded_cols_per_input_block,
             width_wise_output_block_start_index,
             num_cols_already_processed_in_first_output_block});

        // Compute run-time args
        if (full_compute_idx >= 0) {
            desc.kernels[full_compute_idx].emplace_runtime_args(core, {num_input_blocks_to_process});
        }

        // Update index of first tile to read
        tile_start_index += num_tiles_per_input_block * num_input_blocks_per_full_core;
    }

    // Run-time args (cliff core)
    // Note: Only applicable if input is interleaved (sharded input will never have a cliff core)
    std::vector<CoreCoord> cliff_cores = corerange_to_cores(cliff_compute_core_range, std::nullopt, is_row_major);
    if (!cliff_cores.empty()) {
        // There should only ever be 0 or 1 cliff cores
        CoreCoord cliff_core = cliff_cores[0];
        uint32_t height_wise_input_block_start_index = full_cores.size() * num_input_blocks_per_full_core;
        uint32_t width_wise_input_block_index = 0;

        // Handle uneven input sharding width wise (writer run-time arg)
        // Note: Since cliff core is only applicable to interleaved input, this core
        // will never process an uneven shard (or any shard for that matter)
        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;

        // Handle uneven input sharding height wise (reader, compute, writer run-time arg)
        // Note: Since cliff core is only applicable to interleaved input, this core
        // will never process an uneven shard (or any shard for that matter)
        uint32_t num_input_blocks_to_process = num_input_blocks_per_cliff_core;

        // Writer run-time args
        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        writer_ref.emplace_runtime_args(
            cliff_core,
            {dst_buffer,
             num_input_blocks_to_process,
             height_wise_input_block_start_index,
             num_unpadded_cols_per_input_block,
             width_wise_output_block_start_index,
             num_cols_already_processed_in_first_output_block});

        // Reader run-time args (always reading interleaved input as cliff core does not exist for sharded input)
        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        reader_ref.emplace_runtime_args(cliff_core, {src0_buffer, num_tiles_to_read, tile_start_index});

        // Compute run-time args
        if (cliff_compute_idx >= 0) {
            desc.kernels[cliff_compute_idx].emplace_runtime_args(cliff_core, {num_input_blocks_to_process});
        }
    }

    return desc;
}
}  // namespace ttnn::prim
