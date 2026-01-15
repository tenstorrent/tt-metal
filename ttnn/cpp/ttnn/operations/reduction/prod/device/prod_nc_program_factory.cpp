// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "prod_nc_program_factory.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <string>

namespace ttnn::prim {

using namespace tt::constants;

ProdNcProgramFactory::cached_program_t ProdNcProgramFactory::create(
    const ProdNcParams& operation_attributes, const ProdNcInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;
    const int64_t dim = operation_attributes.dim;

    TT_FATAL(dim == 0 || dim == 1, "Dimension ({}) must be either 0 or 1", dim);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = input.device();
    auto program = tt::tt_metal::Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    const auto& input_shape = input.padded_shape();

    [[maybe_unused]] const auto N = input_shape[0];
    const auto C = input_shape[1];
    const auto Ht = input_shape[2] / TILE_HEIGHT;
    const auto Wt = input_shape[3] / TILE_WIDTH;
    const auto HtWt = Ht * Wt;
    const auto CHtWt = C * Ht * Wt;
    const auto num_reduce_input_tile = input_shape[dim];
    const auto input_tile_offset = (dim == 0) ? (CHtWt) : (HtWt);
    const auto num_output_tiles = output.physical_volume() / TILE_HW;

    log_debug(tt::LogTest, "N {} C {} Ht {} Wt {}", N, C, Ht, Wt);
    log_debug(
        tt::LogTest,
        "dim {} num_reduce_input_tile {} input_tile_offset {}, num_output_tiles {}",
        dim,
        num_reduce_input_tile,
        input_tile_offset,
        num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = 2;        // input
    const uint32_t in1_t = 1;        // zero
    const uint32_t intermed0_t = 1;  // accumulated sum
    const uint32_t out0_t = 2;       // output
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    operations::CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {tt::CBIndex::c_0, in0_t},        // input
            {tt::CBIndex::c_1, in1_t},        // zero
            {tt::CBIndex::c_2, intermed0_t},  // accumulated sum
            {tt::CBIndex::c_3, out0_t},       // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(dim)};
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_3;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)cb_id_out};
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/dataflow/reader_prod_nc.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    const auto reader_kernel_id =
        operations::CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id =
        operations::CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1};
    std::map<std::string, std::string> compute_defines;

    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_nc.cpp";
    const auto compute_kernel_1_id = operations::CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_cols_per_core_group_1, compute_args_group_1}, compute_defines);

    std::optional<tt::tt_metal::KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        compute_kernel_2_id = ttnn::operations::CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_reduce_input_tile,
             num_tiles_per_core,
             input_tile_offset,
             tile_offset,
             HtWt,
             CHtWt,
             static_cast<uint32_t>(dim)});

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {output.buffer()->address(),
             num_tiles_per_core,
             tile_offset,
             static_cast<uint32_t>(ttnn::operations::is_dram(output))});

        if (core_group_1.contains(core)) {
            tt::tt_metal::SetRuntimeArgs(
                program, compute_kernel_1_id, core, {num_reduce_input_tile, num_tiles_per_core});
        } else if (core_group_2.contains(core)) {
            TT_FATAL(compute_kernel_2_id.has_value(), "compute_kernel_2_id needs to have a value");
            tt::tt_metal::SetRuntimeArgs(
                program, compute_kernel_2_id.value(), core, {num_reduce_input_tile, num_tiles_per_core});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        tile_offset += num_tiles_per_core;
    }

    return {
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .num_cores_to_be_used = num_cores_to_be_used,
            .num_cores_y = num_cores_y}};
}

void ProdNcProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ProdNcParams& /*operation_attributes*/,
    const ProdNcInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    auto& program = cached_program.program;
    const auto& shared_variables = cached_program.shared_variables;

    const auto* input_buffer = tensor_args.input.buffer();
    const auto* output_buffer = tensor_args.output.buffer();

    for (uint32_t i = 0; i < shared_variables.num_cores_to_be_used; ++i) {
        tt::tt_metal::CoreCoord core = {i / shared_variables.num_cores_y, i % shared_variables.num_cores_y};
        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared_variables.reader_kernel_id, core);
            runtime_args[0] = input_buffer->address();
        }

        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared_variables.writer_kernel_id, core);
            runtime_args[0] = output_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
