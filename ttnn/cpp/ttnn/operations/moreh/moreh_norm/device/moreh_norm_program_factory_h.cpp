// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_norm {

MorehNormOperation::ProgramFactoryH::cached_program_t MorehNormOperation::ProgramFactoryH::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto p = operation_attributes.p;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.get_legacy_shape();
    const auto input_rank = input_shape.rank();

    const auto H = input_shape[-2];
    const auto W = input_shape[-1];

    const auto Ht = H / tt::constants::TILE_HEIGHT;
    const auto Wt = W / tt::constants::TILE_WIDTH;

    const auto num_units = input.volume() / H / W * Wt;

    const auto origin_h = input_shape.without_padding()[-2];

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
    auto [floored_recip_p, recip_p_decimal, recip_p_is_negative] =
        get_floored_p_and_decimal_and_p_is_negative(1.0f / p);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, operation_attributes.compute_kernel_config);

    const auto [num_cores_to_be_used,
                all_cores,
                core_group_1,
                core_group_2,
                num_units_per_core_group_1,
                num_units_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_units);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;

    const uint32_t in0_t{1};  // input
    const uint32_t in1_t{1};  // one
    const uint32_t in2_t{1};  // decimal
    const uint32_t in3_t{1};  // recip_p_decimal
    const uint32_t in4_t{1};  // mask_h

    const uint32_t out0_t{1};  // output

    const uint32_t im0_t{1};  // |x|
    const uint32_t im1_t{1};  // log(|x|)
    const uint32_t im2_t{1};  // exp(log(|x|) * decimal)
    const uint32_t im3_t{1};  // |x|^p
    const uint32_t im4_t{1};  // |x|^p * exp(log(|x|) * decimal) == |x + decimal|^p
    const uint32_t im5_t{1};  // Add(|x + decimal|^p)
    const uint32_t im6_t{1};  // Sum(|x + decimal|^p)

    tt::operations::primary::CreateCircularBuffer(program,
                                                  all_cores,
                                                  cb_data_format,
                                                  {
                                                      {tt::CB::c_in0, in0_t},    // input
                                                      {tt::CB::c_in1, in1_t},    // one
                                                      {tt::CB::c_in2, in2_t},    // decimal
                                                      {tt::CB::c_in3, in3_t},    // recip_p_decimal
                                                      {tt::CB::c_in4, in4_t},    // mask_h
                                                      {tt::CB::c_out0, out0_t},  // output
                                                      {tt::CB::c_intermed0, im0_t, intermed_data_format},
                                                      {tt::CB::c_intermed1, im1_t, intermed_data_format},
                                                      {tt::CB::c_intermed2, im2_t, intermed_data_format},
                                                      {tt::CB::c_intermed3, im3_t, intermed_data_format},
                                                      {tt::CB::c_intermed4, im4_t, intermed_data_format},
                                                      {tt::CB::c_intermed5, im5_t, intermed_data_format},
                                                      {tt::CB::c_intermed6, im6_t, intermed_data_format},
                                                  });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_h/kernels/"
        "reader_moreh_norm_h.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_h/kernels/"
        "writer_moreh_norm_h.cpp";

    const auto reader_kernels_id = tt::operations::primary::CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = tt::operations::primary::CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_h/kernels/"
        "moreh_norm_h_kernel.cpp";

    const auto compute_kernels_id_1 =
        tt::operations::primary::CreateComputeKernel(program,
                                                     compute_kernel_file,
                                                     {core_group_1, num_units_per_core_group_1},
                                                     compute_defines,
                                                     math_fidelity,
                                                     fp32_dest_acc_en,
                                                     math_approx_mode);

    KernelHandle compute_kernels_id_2{0};
    if (!core_group_2.ranges().empty()) {
        compute_kernels_id_2 = tt::operations::primary::CreateComputeKernel(program,
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

        uint32_t num_cols_per_core;
        KernelHandle compute_kernel_id;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_units_per_core_group_1;
            compute_kernel_id = compute_kernels_id_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_units_per_core_group_2;
            compute_kernel_id = compute_kernels_id_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{input.buffer()->address(),
                                                        static_cast<uint32_t>(tt::operations::primary::is_dram(input)),
                                                        *reinterpret_cast<uint32_t*>(&decimal),
                                                        *reinterpret_cast<uint32_t*>(&recip_p_decimal),
                                                        num_cols_per_core,
                                                        tile_offset,
                                                        Ht,
                                                        Wt,
                                                        origin_h};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{output.buffer()->address(),
                                                        static_cast<uint32_t>(tt::operations::primary::is_dram(output)),
                                                        num_cols_per_core,
                                                        tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{num_cols_per_core,
                                                         Ht,
                                                         origin_h,
                                                         floored_p,
                                                         static_cast<uint32_t>(p_is_negative),
                                                         floored_recip_p,
                                                         static_cast<uint32_t>(recip_p_is_negative)};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset += num_cols_per_core;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void MorehNormOperation::ProgramFactoryH::override_runtime_arguments(cached_program_t& cached_program,
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
