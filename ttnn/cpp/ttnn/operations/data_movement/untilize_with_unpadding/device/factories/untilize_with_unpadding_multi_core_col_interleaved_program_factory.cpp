// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_col_interleaved_program_factory.hpp"

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory::cached_program_t
UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory::create(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t num_blocks = input_shape[-1] / TILE_WIDTH;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_tiles_per_col = a.padded_shape()[-2] / TILE_HEIGHT;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t unpadded_row_size_bytes;

    uint32_t el_size;
    if (a.dtype() == DataType::BFLOAT8_B) {
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
        el_size = a.element_size();
    }

    create_cb(tt::CBIndex::c_0, program, all_cores, input_single_tile_size, num_tiles_per_col, input_cb_data_format);
    create_cb(tt::CBIndex::c_16, program, all_cores, output_single_tile_size, num_tiles_per_col, output_cb_data_format);

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

    std::vector<uint32_t> reader_compile_time_args = {num_tiles_2d, third_dim, nblocks_per_core};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_col_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    // writer
    uint32_t total_num_rows = output.logical_shape()[-2];

    std::vector<uint32_t> writer_ct_args = {total_num_rows, ncores, third_dim, TILE_WIDTH, unpadded_row_size_bytes};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_col_multicore.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // compute

    std::string compute_kernel("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_w.cpp");

    if (!core_range.empty()) {
        CreateKernel(
            program,
            compute_kernel,
            core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {nblocks_per_core, num_tiles_per_col, third_dim}});
    }
    if (has_cliff) {
        CreateKernel(
            program,
            compute_kernel,
            core_range_cliff,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {nblocks_per_core_cliff, num_tiles_per_col, third_dim}});
    }

    // RUNTIME ARGS
    const auto& cores = corerange_to_cores(available_grid);
    uint32_t number_blocks_per_core;
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];

        if (has_cliff && i == ncores - 1) {
            number_blocks_per_core = nblocks_per_core_cliff;
        } else {
            number_blocks_per_core = nblocks_per_core;
        }
        uint32_t size_per_row_per_block = nblocks_per_core * TILE_WIDTH * el_size;

        //  writer runtime args
        std::vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            i,
            size_per_row_per_block,
            number_blocks_per_core,
            TILE_WIDTH * el_size,
        };

        // reader runtime args
        const std::array reader_rt_args = {src0_buffer->address(), i, num_tiles_per_row, number_blocks_per_core};
        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = unary_reader_kernel_id,
            .writer_kernel_id = unary_writer_kernel_id,
            .cores = cores,
            .ncores = ncores}};
}

void UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const UntilizeWithUnpaddingParams& /*operation_attributes*/,
    const Tensor& input,
    const Tensor& output) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;
    const auto& ncores = shared_vars.ncores;
    const auto& cores = shared_vars.cores;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    auto& reader_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);

    for (uint32_t i = 0; i < ncores; ++i) {
        const CoreCoord& core = cores[i];
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
