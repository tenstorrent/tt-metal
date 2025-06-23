// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_norm/device/moreh_norm_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_norm {

MorehNormOperation::ProgramFactoryNCOther::cached_program_t MorehNormOperation::ProgramFactoryNCOther::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto dim = operation_attributes.dim;
    const auto p = operation_attributes.p;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();
    const auto input_rank = static_cast<decltype(dim)>(input_shape.rank());

    const auto H = input_shape[-2];
    const auto W = input_shape[-1];

    const auto Ht = H / tt::constants::TILE_HEIGHT;
    const auto Wt = W / tt::constants::TILE_WIDTH;

    const auto num_reduced_tiles_along_dim = input_shape[dim];
    const auto num_output_tiles = output.physical_volume() / tt::constants::TILE_HW;

    uint32_t outer_stride{1};
    for (int64_t j = dim; j < input_rank; ++j) {
        outer_stride *= input_shape[j];
    }
    outer_stride /= tt::constants::TILE_HW;

    uint32_t num_inner_tiles{1};
    for (int64_t j = dim + 1; j < input_rank; ++j) {
        num_inner_tiles *= input_shape[j];
    }
    num_inner_tiles /= tt::constants::TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, operation_attributes.compute_kernel_config);

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_units_per_core_group_1,
         num_units_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;

    const uint32_t in0_t{1};  // input
    const uint32_t in1_t{1};  // one

    const uint32_t out0_t{1};  // output

    const uint32_t im0_t{1};  // f(x)
    const uint32_t im1_t{1};  // calculate f(x) over dimensions

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {tt::CBIndex::c_0, in0_t},    // input
            {tt::CBIndex::c_1, in1_t},    // one
            {tt::CBIndex::c_16, out0_t},  // output
            {tt::CBIndex::c_24, im0_t, intermed_data_format},
            {tt::CBIndex::c_25, im1_t, intermed_data_format},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/"
        "reader_moreh_norm_nc.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/"
        "writer_moreh_norm_nc.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    if (p == 0.0) {
        compute_defines["IS_ZERO"] = "1";
    } else {
        if (p == -std::numeric_limits<float>::infinity()) {
            compute_defines["MINUS_INF"] = "1";
        }
    }

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/"
        "moreh_norm_nc_kernel.cpp";

    const auto compute_kernels_id_1 = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_units_per_core_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    KernelHandle compute_kernels_id_2{0};
    if (!core_group_2.ranges().empty()) {
        compute_kernels_id_2 = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_units_per_core_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core;
        KernelHandle compute_kernel_id;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_units_per_core_group_1;
            compute_kernel_id = compute_kernels_id_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_units_per_core_group_2;
            compute_kernel_id = compute_kernels_id_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input.buffer()->address(),
            static_cast<uint32_t>(is_dram(input)),
            num_output_tiles_per_core,
            tile_offset,
            outer_stride,
            num_inner_tiles,
            num_reduced_tiles_along_dim};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            output.buffer()->address(), static_cast<uint32_t>(is_dram(output)), num_output_tiles_per_core, tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{
            num_output_tiles_per_core,
            num_reduced_tiles_along_dim,
        };
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset += num_output_tiles_per_core;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void MorehNormOperation::ProgramFactoryNCOther::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto& writer_kernels_id = cached_program.shared_variables.writer_kernels_id;
    auto& num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    for (uint32_t icore = 0; icore < num_cores_to_be_used; icore++) {
        CoreCoord core = {icore / num_cores_y, icore % num_cores_y};
        // readers
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
            runtime_args[0] = tensor_args.input.buffer()->address();
        }

        // writer
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
            runtime_args[0] = output.buffer()->address();
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_norm
