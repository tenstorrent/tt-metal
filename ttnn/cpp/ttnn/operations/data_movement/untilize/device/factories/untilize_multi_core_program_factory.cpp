// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include "untilize_multi_core_program_factory.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

UntilizeMultiCoreProgramFactory::cached_program_t UntilizeMultiCoreProgramFactory::create(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& output) {
    tt::tt_metal::Program program{};

    const auto& a = tensor_args.input;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& use_pack_untilize = operation_attributes.use_pack_untilize;
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
    bool has_ordered_cores_with_data = false;

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
    uint32_t num_shards = 0;
    uint32_t num_shards_height = 0;
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

        num_shards_height = tt::div_up(tensor_height, input_shard_height);
        num_shards = num_shards_height * num_input_blocks_across_width;
        if (num_compute_cores >
            num_shards) {  // If the user specified more compute cores than there are data, we need to figure out which
                           // cores have data on them and only activate those cores. To do this, we use information
                           // encoded in the buffer distribution spec.
            if (a.buffer()->buffer_distribution_spec().has_value()) {  // If the tensor also has an nd_shard_spec, then
                                                                       // it has a bufferdistributionspec. Use it.
                auto buffer_dist_spec = a.buffer()->buffer_distribution_spec().value();
                ordered_cores_with_data = buffer_dist_spec.cores_with_data();
                has_ordered_cores_with_data = true;
                compute_core_range = CoreRangeSet(tt::stl::Span<const CoreCoord>(buffer_dist_spec.cores_with_data()));
            } else {  // If the tensor does not have an nd_shard_spec, then we need to create a bufferdistributionspec
                      // from the shard_spec.
                auto buffer_dist_spec = BufferDistributionSpec::from_shard_spec(
                    a.padded_shape(),
                    Shape({input_shard_height, input_shard_width}),
                    a.tensor_spec().tile().get_tile_shape(),
                    input_shard_spec.grid,
                    input_shard_spec.orientation,
                    a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED
                        ? ShardDistributionStrategy::GRID_2D
                        : ShardDistributionStrategy::ROUND_ROBIN_1D);
                ordered_cores_with_data = buffer_dist_spec.cores_with_data();
                has_ordered_cores_with_data = true;
                compute_core_range = CoreRangeSet(tt::stl::Span<const CoreCoord>(buffer_dist_spec.cores_with_data()));
            }

            full_compute_core_range = compute_core_range;
        } else {
            compute_core_range = input_shard_spec.grid;
            full_compute_core_range = input_shard_spec.grid;
        }
        cliff_compute_core_range = CoreRangeSet();
    }

    // Input CB
    uint32_t input_cb_num_tiles;
    if (input_is_sharded) {
        // Have compute core untilize the entire shard at once
        input_cb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core;
    } else {
        if (num_input_blocks_per_full_core == 1) {
            // No need to double buffer if the core is only processing a single block
            input_cb_num_tiles = num_tiles_per_input_block;
        } else {
            // Double buffer if the core is processing 2+ blocks
            input_cb_num_tiles = num_tiles_per_input_block * 2;
        }
    }
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0,
        program,
        compute_core_range,
        input_single_tile_size,
        input_cb_num_tiles,
        input_cb_data_format,
        input_is_sharded ? src0_buffer : nullptr);

    // Output CB
    uint32_t output_cb_num_tiles;
    if (num_input_blocks_per_full_core == 1) {
        // No need to double buffer if the core is only processing a single block
        output_cb_num_tiles = num_tiles_per_input_block;
    } else {
        // Double buffer if the core is processing 2+ blocks
        output_cb_num_tiles = num_tiles_per_input_block * 2;
    }
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        compute_core_range,
        output_single_tile_size,
        output_cb_num_tiles,
        output_cb_data_format);

    // Reader compile-time args and kernel
    KernelHandle unary_reader_kernel_id;
    if (input_is_sharded) {
        // Sharded input
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};
        unary_reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
            compute_core_range,
            tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    } else {
        // Interleaved input
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
        unary_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
            "reader_unary_start_id.cpp",
            compute_core_range,
            ReaderDataMovementConfig(reader_compile_time_args));
    }

    // Writer compile-time args
    uint32_t output_element_size = output.element_size();
    uint32_t output_page_width =
        tensor_width;  // In height-sharded and interleaved cases, the output page is the entire tensor row
    uint32_t output_num_blocks_across_width = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
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

    // Writer kernel
    KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multi_core.cpp",
        compute_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    // Compute kernel file
    std::string compute_kernel;
    if (!use_pack_untilize || a.dtype() == DataType::UINT16) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel = std::string(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp");
        unpack_to_dest_mode[src0_cb_index] =
            UnpackToDestMode::Default;  // TODO: We need SFPU untilize for FP32 (#30400, #33795)
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
        compute_kernel = std::string(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
            "pack_untilize_variable_num_blocks.cpp");
    }

    // Compute compile-time args and kernel
    // Note: This condition is always true for sharded input
    KernelHandle untilize_kernel_id = 0;
    std::map<std::string, std::string> compute_kernel_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_kernel_defines["DST_ACCUM_MODE"] = "1";
    }
    if (!full_compute_core_range.ranges().empty()) {
        std::vector<uint32_t> compute_compile_time_args = {
            (uint32_t)num_tiles_per_input_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};
        untilize_kernel_id = CreateKernel(
            program,
            compute_kernel,
            full_compute_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_compile_time_args,
                .defines = compute_kernel_defines});
    }

    // Compute Cliff compile_time args and kernel
    // Note: This condition is always false for sharded input (sharded input will never have a cliff core)
    KernelHandle untilize_cliff_kernel_id = 0;
    if (!cliff_compute_core_range.ranges().empty()) {
        std::vector<uint32_t> compute_compile_time_args_cliff = {
            (uint32_t)num_tiles_per_input_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};
        untilize_cliff_kernel_id = CreateKernel(
            program,
            compute_kernel,
            cliff_compute_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_compile_time_args_cliff,
                .defines = compute_kernel_defines});
    }

    // Run-time arg assignment
    // Note: This variable is only applicable to interleaved input
    uint32_t tile_start_index = 0;

    // Run-time args (full cores)
    // Note: For sharded input, these are the only cores used
    bool is_row_major = input_is_sharded ? a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR : true;
    std::vector<CoreCoord> full_cores = has_ordered_cores_with_data
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
        std::vector<uint32_t> reader_run_time_args;
        if (input_is_sharded) {
            // Sharded input
            reader_run_time_args = {num_tiles_to_read};
        } else {
            // Interleaved input
            reader_run_time_args = {
                src0_buffer->address(),
                num_tiles_to_read,
                tile_start_index,
            };
        }

        // Writer run-time args
        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        std::vector<uint32_t> writer_run_time_args = {
            dst_buffer->address(),
            num_input_blocks_to_process,
            height_wise_input_block_start_index,
            num_unpadded_cols_per_input_block,
            width_wise_output_block_start_index,
            num_cols_already_processed_in_first_output_block};

        // Compute run-time args
        std::vector<uint32_t> compute_run_time_args = {num_input_blocks_to_process};

        // Set run-time arg
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, untilize_kernel_id, core, compute_run_time_args);

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
        std::vector<uint32_t> writer_run_time_args = {
            dst_buffer->address(),
            num_input_blocks_to_process,
            height_wise_input_block_start_index,
            num_unpadded_cols_per_input_block,
            width_wise_output_block_start_index,
            num_cols_already_processed_in_first_output_block};

        // Reader run-time args (always reading interleaved input as cliff core does not exist for sharded input)
        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        std::vector<uint32_t> reader_run_time_args = {
            src0_buffer->address(),
            num_tiles_to_read,
            tile_start_index,
        };

        // Compute run-time args
        std::vector<uint32_t> compute_run_time_args = {num_input_blocks_to_process};

        // Set run-time args
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, cliff_core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, cliff_core, writer_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, untilize_cliff_kernel_id, cliff_core, compute_run_time_args);
    }

    std::vector<CoreCoord> cores_with_run_time_args;
    cores_with_run_time_args.reserve(full_cores.size() + cliff_cores.size());
    cores_with_run_time_args.insert(cores_with_run_time_args.end(), full_cores.begin(), full_cores.end());
    cores_with_run_time_args.insert(cores_with_run_time_args.end(), cliff_cores.begin(), cliff_cores.end());

    return cached_program_t{
        std::move(program),
        {unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cb_output, cores_with_run_time_args}};
}

void UntilizeMultiCoreProgramFactory::override_runtime_arguments(
    UntilizeMultiCoreProgramFactory::cached_program_t& cached_program,
    const UntilizeOperationAttributes& /*operation_attributes*/,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cb_src0 = cached_program.shared_variables.cb_src0;
    auto& cores_with_runtime_args = cached_program.shared_variables.cores_with_runtime_args;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    bool input_is_sharded = tensor_args.input.is_sharded();

    // Reader
    if (input_is_sharded) {
        // Sharded input
        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    } else {
        // Interleaved input
        auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
        for (const CoreCoord& core : cores_with_runtime_args) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_buffer->address();
        }
    }

    // Writer
    auto& runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
    for (const CoreCoord& core : cores_with_runtime_args) {
        auto& runtime_args = runtime_args_by_core[core.x][core.y];
        runtime_args[0] = dst_buffer->address();
    }
}
}  // namespace ttnn::prim
