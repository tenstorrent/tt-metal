// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_clip_grad_norm_step1_device_operation.hpp"
#include <tt-metalium/assert.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step1 {

std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative(float ord) {
    auto p = std::floor(ord);
    auto decimal = ord - p;
    const bool p_is_negative = p < 0.0f;
    if (p_is_negative) {
        p = -p;
    }
    return std::make_tuple(static_cast<uint32_t>(p), decimal, p_is_negative);
}

MorehClipGradNormStep1Operation::ProgramFactory::cached_program_t
MorehClipGradNormStep1Operation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tmp_pow_sum) {
    auto& inputs = tensor_args.inputs;
    auto norm_type = operation_attributes.norm_type;
    auto tile_offset_of_tmp_pow_sum = operation_attributes.tile_offset_of_tmp_pow_sum;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = tmp_pow_sum.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_inputs = static_cast<uint32_t>(inputs.size());

    std::vector<std::pair<uint32_t, uint32_t>> origin_hw_vec;
    origin_hw_vec.reserve(num_inputs);

    for (uint32_t j = 0; j < num_inputs; ++j) {
        const auto& input_shape_without_padding = inputs.at(j).logical_shape();
        origin_hw_vec.emplace_back(input_shape_without_padding[2], input_shape_without_padding[3]);
    }

    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(norm_type);

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
    const uint32_t in0_t = 1;  // input(==x)
    const uint32_t in1_t = 1;  // one
    const uint32_t in2_t = 1;  // decimal
    const uint32_t in3_t = 2;  // mask_h_w

    const uint32_t out0_t = 1;  // output(==y)

    const uint32_t im0_t = 1;  // |x|
    const uint32_t im1_t = 1;  // |x|^p
    const uint32_t im2_t = 1;  // Add[|x|^p * exp(log(|x|) * decimal)]
    const uint32_t im3_t = 1;  // log(|x|)
    const uint32_t im4_t = 1;  // exp(log(|x|) * decimal)
    const uint32_t im5_t = 1;  // |x|^p * exp(log(|x|) * decimal)

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tmp_pow_sum.dtype());

    CreateCircularBuffer(
        program,
        core_group_1,
        cb_data_format,
        {
            {tt::CBIndex::c_0, in0_t},    // input(==x)
            {tt::CBIndex::c_1, in1_t},    // one
            {tt::CBIndex::c_2, in2_t},    // decimal
            {tt::CBIndex::c_3, in3_t},    // mask_h_w
            {tt::CBIndex::c_16, out0_t},  // output(==y)
            {tt::CBIndex::c_24, im0_t},   // |x|
            {tt::CBIndex::c_25, im1_t},   // |x|^p
            {tt::CBIndex::c_26, im2_t},   // Add[|x|^p * exp(log(|x|) * decimal)]
            {tt::CBIndex::c_27, im3_t},   // log(|x|)
            {tt::CBIndex::c_28, im4_t},   // exp(log(|x|) * decimal)
            {tt::CBIndex::c_29, im5_t},   // |x|^p * exp(log(|x|) * decimal)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/"
        "reader_moreh_clip_grad_norm_step1.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/"
        "writer_moreh_clip_grad_norm_step1.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, core_group_1);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, core_group_1);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/"
        "moreh_clip_grad_norm_step1_kernel.cpp";

    const auto compute_kernel_id =
        CreateComputeKernel(program, compute_kernel_file, {core_group_1, num_inputs_per_core_group_1}, compute_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////

    const auto output_addr = tmp_pow_sum.buffer()->address();
    auto cores = grid_to_cores(num_cores_to_be_used, num_cores_x, num_cores_y, false);

    uint32_t tile_offset = tile_offset_of_tmp_pow_sum;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);

        const auto& input = inputs.at(i);
        const auto input_addr = input.buffer()->address();
        const auto num_tiles = static_cast<uint32_t>(input.physical_volume()) / tt::constants::TILE_HW;
        const auto [origin_h, origin_w] = origin_hw_vec.at(i);

        // reader
        const std::array reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(input.buffer()->is_dram()),
            num_tiles,
            *reinterpret_cast<uint32_t*>(&decimal),
            origin_h,
            origin_w};
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        // writer
        const std::array writer_runtime_args{
            output_addr, static_cast<uint32_t>(tmp_pow_sum.buffer()->is_dram()), tile_offset};
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        // compute
        const std::array compute_runtime_args{
            num_tiles,
            p,
            static_cast<uint32_t>(p_is_negative),
            origin_h,
            origin_w,
        };
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset++;
    }

    return {
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, num_cores_to_be_used, num_cores_y}};
}

void MorehClipGradNormStep1Operation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tmp_pow_sum) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;
    auto num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    const auto norm_type = operation_attributes.norm_type;
    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(norm_type);
    auto output_buffer = tmp_pow_sum.buffer();
    const auto output_address = output_buffer->address();

    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = tensor_args.inputs.at(i).buffer()->address();
            runtime_args[3] = *reinterpret_cast<uint32_t*>(&decimal);
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_address;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, compute_kernel_id, core);
            runtime_args[1] = p;
            runtime_args[2] = static_cast<uint32_t>(p_is_negative);
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step1
