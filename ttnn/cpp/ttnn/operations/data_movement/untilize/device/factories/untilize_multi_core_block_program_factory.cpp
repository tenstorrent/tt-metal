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
#include "untilize_multi_core_block_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

UntilizeMultiCoreBlockProgramFactory::cached_program_t UntilizeMultiCoreBlockProgramFactory::create(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    tt::tt_metal::Program program{};
    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& use_pack_untilize = operation_attributes.use_pack_untilize;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    uint32_t a_tile_width = a.tensor_spec().tile().get_width();
    uint32_t a_tile_height = a.tensor_spec().tile().get_height();

    uint32_t num_tiles_per_row = a.padded_shape()[-1] / a_tile_width;
    uint32_t num_tiles_per_col = a.padded_shape()[-2] / a_tile_height;

    uint32_t num_blocks = (a.padded_shape()[-1] * a.padded_shape()[-2]) / (a_tile_height * a_tile_width);

    auto
        [ncores,
         all_cores,
         core_range,
         cliff_row_core_range,
         cliff_col_core_range,
         cliff_col_row_core_range,
         nblocks_per_core,
         single_block_size,
         single_block_size_cliff_row,
         single_block_size_cliff_col,
         has_cliff_row,
         has_cliff_col,
         full_cores_per_row,
         full_cores_per_col] =
            ttnn::split_blocks_for_tilize_wh(grid_size, num_blocks, num_tiles_per_row, num_tiles_per_col);

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);
    uint32_t row_size_bytes;

    uint32_t el_size = a.element_size();
    if (a.dtype() == DataType::BFLOAT8_B) {
        row_size_bytes = input_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        row_size_bytes = input_shape[-1] * a.element_size();
    }

    if (!core_range.empty()) {
        create_cb(
            tt::CBIndex::c_0, program, core_range, input_single_tile_size, single_block_size, input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16, program, core_range, output_single_tile_size, single_block_size, output_cb_data_format);
    }

    if (has_cliff_col && has_cliff_row) {
        create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_col_row_core_range,
            input_single_tile_size,
            single_block_size_cliff_row,
            input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16,
            program,
            cliff_col_row_core_range,
            output_single_tile_size,
            single_block_size_cliff_row,
            output_cb_data_format);
    }

    if (has_cliff_row) {
        create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_row_core_range,
            input_single_tile_size,
            single_block_size_cliff_row,
            input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16,
            program,
            cliff_row_core_range,
            output_single_tile_size,
            single_block_size_cliff_row,
            output_cb_data_format);
    }

    if (has_cliff_col) {
        create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_col_core_range,
            input_single_tile_size,
            single_block_size,
            input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16,
            program,
            cliff_col_core_range,
            output_single_tile_size,
            single_block_size,
            output_cb_data_format);
    }

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // reader

    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    std::vector<uint32_t> reader_compile_time_args = {num_tiles_2d, third_dim, total_tiles_per_row};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    // writer
    uint32_t total_num_rows = output.logical_shape()[-2];
    std::vector<uint32_t> writer_ct_args = {total_num_rows, third_dim, TILE_HEIGHT, row_size_bytes};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_wh_multicore.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // compute
    bool use_pack_kernel = true;
    if (!use_pack_untilize || a.dtype() == DataType::UINT16 ||
        (a.dtype() == DataType::FLOAT32 && num_tiles_per_row > MAX_PACK_UNTILIZE_WIDTH)) {
        use_pack_kernel = false;
    }
    if (!core_range.empty()) {
        CreateKernel(
            program,
            use_pack_kernel
                ? "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_wh.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp",
            core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {single_block_size, single_block_size, third_dim}});
    }
    if (has_cliff_col && has_cliff_row) {
        CreateKernel(
            program,
            use_pack_kernel
                ? "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_wh.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp",
            cliff_col_row_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {single_block_size_cliff_col, single_block_size_cliff_row, third_dim}});
    }
    if (has_cliff_row) {
        CreateKernel(
            program,
            use_pack_kernel
                ? "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_wh.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp",
            cliff_row_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {single_block_size, single_block_size_cliff_row, third_dim}});
    }

    if (has_cliff_col) {
        CreateKernel(
            program,
            use_pack_kernel
                ? "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_wh.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp",
            cliff_col_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {single_block_size_cliff_col, single_block_size, third_dim}});
    }

    // RUNTIME ARGS
    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    uint32_t start_row_id = 0;
    uint32_t start_column_id = 0;
    uint32_t tile_start_id = 0;
    uint32_t single_block_size_row_arg;
    uint32_t single_block_size_col_arg;

    uint32_t total_row_cores = full_cores_per_row;
    if (has_cliff_row) {
        total_row_cores++;
    }
    uint32_t cores_col_count = 1;

    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];

        if (has_cliff_col && has_cliff_row && i == ncores - 1) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size_cliff_col;

        } else if (has_cliff_row && i != 0 && ((i + 1) % (full_cores_per_row + 1)) == 0) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size;

        } else if (i < total_row_cores * full_cores_per_col) {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size;

        } else {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size_cliff_col;
        }

        //  writer runtime args
        std::vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            TILE_WIDTH * el_size * single_block_size_row_arg,
            start_row_id,
            start_column_id,
            single_block_size_row_arg,
            single_block_size_col_arg,
        };

        // reader runtime args
        const std::array reader_rt_args = {
            src0_buffer->address(), tile_start_id, single_block_size_row_arg, single_block_size_col_arg};
        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * TILE_WIDTH * el_size);
        start_column_id = end_column_id % row_size_bytes;
        if (end_column_id % row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * TILE_HEIGHT;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    return UntilizeMultiCoreBlockProgramFactory::cached_program_t{
        std::move(program),
        UntilizeMultiCoreBlockProgramFactory::shared_variables_t{
            .reader_kernel_id = unary_reader_kernel_id,
            .writer_kernel_id = unary_writer_kernel_id,
            .cores_with_runtime_args = cores}};
}

void UntilizeMultiCoreBlockProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const UntilizeOperationAttributes& /*operation_attributes*/,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores_with_runtime_args;
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    auto& reader_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(cached_program.program, reader_kernel_id);
    auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(cached_program.program, writer_kernel_id);

    for (const auto& core : cores) {
        {
            auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_buffer->address();
        }
        {
            auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    }
}
}  // namespace ttnn::prim
