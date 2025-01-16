// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_cumsum {

MorehCumsumDeviceOperation::ProgramFactory::cached_program_t MorehCumsumDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& input = tensor_args.input;

    auto dim = operation_attributes.dim;
    auto flip = operation_attributes.flip;

    TT_ASSERT(dim == 0 || dim == 1);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = input.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output.get_dtype());

    const auto& input_shape = input.get_legacy_shape();
    const auto& input_shape_without_padding = input_shape.without_padding();

    const auto N = input_shape[0];
    const auto C = input_shape[1];
    const auto Ht = input_shape[2] / tt::constants::TILE_HEIGHT;
    const auto Wt = input_shape[3] / tt::constants::TILE_WIDTH;
    const auto HtWt = Ht * Wt;
    const auto CHtWt = C * HtWt;
    const auto NHtWt = N * HtWt;
    const auto num_cumsum_tiles = input_shape[dim];
    const auto input_tile_offset = (dim == 0) ? (CHtWt) : (HtWt);
    const auto num_tiles_per_chip = (dim == 0) ? (CHtWt) : (NHtWt);

    log_debug(tt::LogOp, "N {} C {} Ht {} Wt {}", N, C, Ht, Wt);
    log_debug(
        tt::LogOp,
        "dim {} num_cumsum_tiles {} input_tile_offset {} num_tiles_per_chip {}",
        dim,
        num_cumsum_tiles,
        input_tile_offset,
        num_tiles_per_chip);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = 2;        // input
    const uint32_t in1_t = 1;        // zero
    const uint32_t intermed0_t = 1;  // accumulated sum
    const uint32_t out0_t = 2;       // output

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, operation_attributes.compute_kernel_config);

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_tiles_per_chip);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {tt::CBIndex::c_0, in0_t},         // input
            {tt::CBIndex::c_1, in1_t},         // zero
            {tt::CBIndex::c_24, intermed0_t},  // accumulated sum
            {tt::CBIndex::c_16, out0_t},       // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_cumsum/device/kernels/reader_moreh_cumsum_nc.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_cumsum/device/kernels/writer_moreh_cumsum_nc.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1};
    std::map<string, string> compute_defines;
    const auto compute_kernel_file = "ttnn/cpp/ttnn/operations/moreh/moreh_cumsum/device/kernels/moreh_cumsum_nc.cpp";
    const auto compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_cols_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             num_cumsum_tiles,
             num_tiles_per_core,
             input_tile_offset,
             tile_offset,
             static_cast<uint32_t>(is_dram(input)),
             HtWt,
             CHtWt,
             static_cast<uint32_t>(dim),
             flip});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {output.buffer()->address(),
             num_cumsum_tiles,
             num_tiles_per_core,
             input_tile_offset,
             tile_offset,
             static_cast<uint32_t>(is_dram(output)),
             HtWt,
             CHtWt,
             static_cast<uint32_t>(dim),
             flip});

        if (core_group_1.contains(core)) {
            SetRuntimeArgs(program, compute_kernel_1_id, core, {num_cumsum_tiles, num_tiles_per_core});
        } else if (core_group_2.contains(core)) {
            TT_ASSERT(compute_kernel_2_id.has_value());
            SetRuntimeArgs(program, compute_kernel_2_id.value(), core, {num_cumsum_tiles, num_tiles_per_core});
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }
        tile_offset += num_tiles_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, num_cores_y}};
}

void MorehCumsumDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const auto& input = tensor_args.input;

    auto input_address = input.buffer()->address();
    auto output_address = output.buffer()->address();
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = input_address;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = output_address;
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_cumsum
