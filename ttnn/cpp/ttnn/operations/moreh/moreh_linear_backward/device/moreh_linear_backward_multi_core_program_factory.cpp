// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_linear_backward_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {

MorehBiasAddBackwardOperation::MultiCoreProgramFactory::cached_program_t
MorehBiasAddBackwardOperation::MultiCoreProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& bias_grad) {
    using namespace tt;
    using namespace tt::tt_metal;

    Program program{};
    auto& output_grad = tensor_args.output_grad;

    const auto& bias_grad_shape = bias_grad.logical_shape();
    const auto& output_grad_shape_wo_padding = output_grad.logical_shape();

    auto bias_grad_memory_config = operation_attributes.bias_grad_memory_config;
    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    const bool do_mask_h = (output_grad_shape_wo_padding[-2] % constants::TILE_HEIGHT) != 0;
    const uint32_t mask_h =
        do_mask_h ? output_grad_shape_wo_padding[-2] % constants::TILE_HEIGHT : constants::TILE_HEIGHT;
    const bool do_mask_w = (output_grad_shape_wo_padding[-1] % constants::TILE_WIDTH) != 0;
    const uint32_t mask_w =
        do_mask_w ? output_grad_shape_wo_padding[-1] % constants::TILE_WIDTH : constants::TILE_WIDTH;

    const auto& output_grad_shape = output_grad.padded_shape();
    uint32_t batch_num = output_grad.physical_volume() / output_grad_shape[-2] / output_grad_shape[-1];
    uint32_t Ht = output_grad_shape[-2] / constants::TILE_HEIGHT;
    uint32_t Wt = output_grad_shape[-1] / constants::TILE_WIDTH;
    uint32_t num_tiles = batch_num * Ht;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    IDevice* device = output_grad.device();
    auto grid = device->compute_with_storage_grid_size();
    auto arch = device->arch();
    const auto num_cores_y = grid.y;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = split_work_to_cores(grid, Wt);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 2;
    const uint32_t in1_t = 1;
    const uint32_t in2_t = 2;  // mask_h_w

    const uint32_t out0_t = 1;
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;
    auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {{CBIndex::c_0, in0_t},    // output_grad
         {CBIndex::c_1, in1_t},    // scaler
         {CBIndex::c_2, in2_t},    // mask_h_w
         {CBIndex::c_16, out0_t},  // bias_grad
         {CBIndex::c_24, im0_t},
         {CBIndex::c_25, im1_t, (fp32_dest_acc_en) ? tt::DataFormat::Float32 : cb_data_format}});

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const ::bfloat16 bfloat_scaler_value = ::bfloat16(1.0f);
    const uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(output_grad)), packed_scaler_value};
    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(bias_grad))};

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/reader_moreh_bias_backward_h.cpp";

    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/writer_moreh_bias_backward.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1};
    std::map<string, string> compute_defines;
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
        unpack_to_dest_mode[tt::CBIndex::c_25] = UnpackToDestMode::UnpackToDestFp32;
    }
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/moreh_bias_backward_multi_core_h.cpp";

    const auto compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_cols_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        unpack_to_dest_mode);

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
            math_approx_mode,
            unpack_to_dest_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_cols_per_core = 0;
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        bool core_has_last_wt = (tile_offset + num_cols_per_core == Wt) ? (true) : (false);
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {output_grad.buffer()->address(),
             num_tiles,
             Wt,
             num_cols_per_core,
             tile_offset,
             mask_h,
             mask_w,
             static_cast<uint32_t>(do_mask_h),
             static_cast<uint32_t>(do_mask_w && core_has_last_wt)});

        SetRuntimeArgs(
            program, writer_kernel_id, core, {bias_grad.buffer()->address(), num_cols_per_core, tile_offset});

        if (core_group_1.contains(core)) {
            SetRuntimeArgs(
                program,
                compute_kernel_1_id,
                core,
                {batch_num,
                 Ht,
                 num_cols_per_core,  // Wt_per_core
                 static_cast<uint32_t>(do_mask_h),
                 static_cast<uint32_t>(do_mask_w && core_has_last_wt)});
        } else if (core_group_2.contains(core)) {
            TT_ASSERT(compute_kernel_2_id.has_value());
            SetRuntimeArgs(
                program,
                compute_kernel_2_id.value(),
                core,
                {batch_num,
                 Ht,
                 num_cols_per_core,  // Wt_per_core
                 static_cast<uint32_t>(do_mask_h),
                 static_cast<uint32_t>(do_mask_w && core_has_last_wt)});
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }
        tile_offset += num_cols_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y}};
}

void MorehBiasAddBackwardOperation::MultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    auto output_grad_buffer = tensor_args.output_grad.buffer();
    auto bias_grad_buffer = tensor_return_value.buffer();

    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = output_grad_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = bias_grad_buffer->address();
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_linear_backward
