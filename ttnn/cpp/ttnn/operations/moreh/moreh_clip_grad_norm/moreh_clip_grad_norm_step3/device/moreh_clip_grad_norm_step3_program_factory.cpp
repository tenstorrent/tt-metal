// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_clip_grad_norm_step3_device_operation.hpp"
#include <tt-metalium/assert.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step3 {

std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative(float ord) {
    auto p = std::floor(ord);
    auto decimal = ord - p;
    const bool p_is_negative = p < 0.0f;
    if (p_is_negative) {
        p = -p;
    }
    return std::make_tuple(static_cast<uint32_t>(p), decimal, p_is_negative);
}

MorehClipGradNormStep3Operation::ProgramFactory::cached_program_t
MorehClipGradNormStep3Operation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& inputs) {
    auto& clip_coef_clamped = tensor_args.clip_coef_clamped;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = inputs.at(0).device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_inputs = static_cast<uint32_t>(inputs.size());

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_inputs_per_core_group_1,
         num_inputs_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_inputs);
    TT_FATAL(core_group_2.ranges().empty(), "core_group_2 must be empty");
    TT_FATAL(num_inputs_per_core_group_1 == 1, "num_inputs_per_core_group_1 must be 1");
    TT_FATAL(num_inputs_per_core_group_2 == 0, "num_inputs_per_core_group_2 must be 0");

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;  // input(inplace)
    const uint32_t in1_t = 1;  // clip_coef_clamped

    const uint32_t out0_t = 1;  // output(inplace)

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(inputs.at(0).get_dtype());

    CreateCircularBuffer(
        program,
        core_group_1,
        cb_data_format,
        {
            {tt::CBIndex::c_0, in0_t},    // input(inplace)
            {tt::CBIndex::c_1, in1_t},    // clip_coef_clamped
            {tt::CBIndex::c_16, out0_t},  // output(inplace)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/"
        "reader_moreh_clip_grad_norm_step3.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/"
        "writer_moreh_clip_grad_norm_step3.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, core_group_1);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, core_group_1);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/"
        "moreh_clip_grad_norm_step3_kernel.cpp";

    const auto compute_kernel_id =
        CreateComputeKernel(program, compute_kernel_file, {core_group_1, num_inputs_per_core_group_1});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto cores = grid_to_cores(num_cores_to_be_used, num_cores_x, num_cores_y, false);
    const auto clip_coef_clamped_addr = clip_coef_clamped.buffer()->address();
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);

        const auto& input = inputs.at(i);
        const auto input_addr = input.buffer()->address();
        const auto num_tiles = input.volume() / tt::constants::TILE_HW;

        // reader
        const std::array reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(input.buffer()->is_dram()),
            clip_coef_clamped_addr,
            static_cast<uint32_t>(clip_coef_clamped.buffer()->is_dram()),
            num_tiles};
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        // writer
        const std::array writer_runtime_args{input_addr, static_cast<uint32_t>(input.buffer()->is_dram()), num_tiles};
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        // compute
        const std::array compute_runtime_args{num_tiles};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y}};
}

void MorehClipGradNormStep3Operation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& inputs) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    auto clip_coef_clamped_buffer = tensor_args.clip_coef_clamped.buffer();
    const auto clip_coef_clamped_address = clip_coef_clamped_buffer->address();

    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = inputs.at(i).buffer()->address();
            runtime_args[2] = clip_coef_clamped_address;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = inputs.at(i).buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step3
