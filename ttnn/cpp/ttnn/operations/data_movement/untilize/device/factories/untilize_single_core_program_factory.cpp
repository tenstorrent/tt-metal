// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "untilize_single_core_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {
UntilizeSingleCoreProgramFactory::cached_program_t UntilizeSingleCoreProgramFactory::create(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& use_pack_untilize = operation_attributes.use_pack_untilize;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::tt_metal::Program program{};

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

    bool input_is_sharded = a.memory_config().is_sharded();
    bool output_is_sharded = output.memory_config().is_sharded();

    uint32_t num_tiles = a.physical_volume() / tile_volume;
    uint32_t num_blocks_across_height = a.physical_volume() / a.padded_shape()[-1] / tile_height;
    uint32_t num_columns_of_blocks = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        num_columns_of_blocks = a.padded_shape()[-1] / output.shard_spec().value().shape[1];
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
    uint32_t input_cb_num_tiles = num_tiles_per_block;
    auto [src0_cb_index, cb_src0] =
        create_cb(tt::CBIndex::c_0, program, core, input_single_tile_size, input_cb_num_tiles, input_cb_data_format);

    // Output CB
    uint32_t output_cb_num_tiles = num_tiles_per_block;
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16, program, core, output_single_tile_size, output_cb_num_tiles, output_cb_data_format);

    // Reader compute defines
    std::map<std::string, std::string> reader_compute_defines;
    if (input_is_sharded) {
        reader_compute_defines["SHARDED"] = "1";
    }

    // Writer compute defines
    std::map<std::string, std::string> writer_compute_defines;
    if (output_is_sharded) {
        writer_compute_defines["SHARDED"] = "1";
    }

    // Reader compile-time args
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};
    if (input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(a, reader_compile_time_args);
    } else {
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    }

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "reader_unary_start_id.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_compute_defines));

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
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output, writer_compile_time_args);
    } else {
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    }

    // Untilized writer
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_single_core.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_compute_defines));

    // Compute file path
    std::map<std::string, std::string> compute_kernel_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_kernel_defines["DST_ACCUM_MODE"] = "1";
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }
    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (!use_pack_untilize || a.dtype() == DataType::UINT16 ||
        (a.dtype() == DataType::FLOAT32 && num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH)) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel =
            std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
        unpack_to_dest_mode[src0_cb_index] =
            UnpackToDestMode::Default;  // TODO: We need SFPU untilize for FP32 (#30400, #33795)
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
    }

    // Compute compile-time args
    uint32_t num_blocks = num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height;
    std::vector<uint32_t> compute_compile_time_args = {
        (uint32_t)num_blocks, (uint32_t)num_tiles_per_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};

    // Compute kernel
    tt::tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_compile_time_args,
            .defines = compute_kernel_defines});

    // Reader run-time args
    uint32_t start_page_id = 0;
    std::vector<uint32_t> reader_run_time_args = {
        src0_buffer->address(),
        num_tiles,
        start_page_id,
    };
    if (input_is_sharded) {
        shard_builder::extend_sharding_run_time_args(a, reader_run_time_args);
    }

    // Writer run-time args
    std::vector<uint32_t> writer_run_time_args = {dst_buffer->address()};
    if (output_is_sharded) {
        shard_builder::extend_sharding_run_time_args(output, writer_run_time_args);
    }

    // Set run-time args
    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_run_time_args);
    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_run_time_args);

    return UntilizeSingleCoreProgramFactory::cached_program_t{
        std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id}};
}

void UntilizeSingleCoreProgramFactory::override_runtime_arguments(
    UntilizeSingleCoreProgramFactory::cached_program_t& cached_program,
    const UntilizeOperationAttributes& /*operation_attributes*/,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    CoreCoord core = {0, 0};

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
    }

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        runtime_args[0] = dst_buffer->address();
    }
}
}  // namespace ttnn::prim
