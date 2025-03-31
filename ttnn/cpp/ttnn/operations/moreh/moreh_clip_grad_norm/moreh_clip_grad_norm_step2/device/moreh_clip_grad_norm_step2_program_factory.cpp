// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_clip_grad_norm_step2_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step2 {

std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative(float ord) {
    auto p = std::floor(ord);
    auto decimal = ord - p;
    const bool p_is_negative = p < 0.0f;
    if (p_is_negative) {
        p = -p;
    }
    return std::make_tuple(static_cast<uint32_t>(p), decimal, p_is_negative);
}

MorehClipGradNormStep2Operation::ProgramFactory::cached_program_t
MorehClipGradNormStep2Operation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& total_norm) {
    const auto& tmp_pow_sum = tensor_args.tmp_pow_sum;
    auto norm_type = operation_attributes.norm_type;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = tmp_pow_sum.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_tiles = tmp_pow_sum.volume() / tt::constants::TILE_HW;

    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(1.0f / norm_type);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord single_core = {0, 0};

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;  // input(==tmp_pow_sum)
    const uint32_t in1_t = 1;  // decimal

    // x^p * exp(log(x) * decimal)
    const uint32_t out0_t = 1;  // output(==total_norm)

    const uint32_t im0_t = 1;  // Sum[tmp_pow_sum](==x)
    const uint32_t im1_t = 1;  // x^p
    const uint32_t im2_t = 1;  // log(x)
    const uint32_t im3_t = 1;  // exp(log(x) * decimal)

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(total_norm.get_dtype());

    CreateCircularBuffer(
        program,
        single_core,
        cb_data_format,
        {
            {tt::CBIndex::c_0, in0_t},    // input(==tmp_pow_sum)
            {tt::CBIndex::c_1, in1_t},    // decimal
            {tt::CBIndex::c_16, out0_t},  // output(==total_norm)
            {tt::CBIndex::c_24, im0_t},   // Sum[tmp_pow_sum](==x)
            {tt::CBIndex::c_25, im1_t},   // x^p
            {tt::CBIndex::c_26, im2_t},   // log(x)
            {tt::CBIndex::c_27, im3_t},   // exp(log(x) * decimal)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/kernels/"
        "reader_moreh_clip_grad_norm_step2.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/kernels/"
        "writer_moreh_clip_grad_norm_step2.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, single_core);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, single_core);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/kernels/"
        "moreh_clip_grad_norm_step2_kernel.cpp";

    const auto compute_kernel_id = CreateComputeKernel(program, compute_kernel_file, {single_core, num_tiles});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = tmp_pow_sum.buffer()->address();
    const auto output_addr = total_norm.buffer()->address();

    // reader
    const std::array reader_runtime_args{
        input_addr,
        static_cast<uint32_t>(tmp_pow_sum.buffer()->is_dram()),
        num_tiles,
        *reinterpret_cast<uint32_t*>(&decimal)};
    SetRuntimeArgs(program, reader_kernel_id, single_core, reader_runtime_args);

    // writer
    const std::array writer_runtime_args{output_addr, static_cast<uint32_t>(total_norm.buffer()->is_dram())};
    SetRuntimeArgs(program, writer_kernel_id, single_core, writer_runtime_args);

    // compute
    const std::array compute_runtime_args{num_tiles, p, static_cast<uint32_t>(p_is_negative)};
    SetRuntimeArgs(program, compute_kernel_id, single_core, compute_runtime_args);

    return {std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, single_core}};
}

void MorehClipGradNormStep2Operation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& total_norm) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;
    auto single_core = cached_program.shared_variables.single_core;

    const auto norm_type = operation_attributes.norm_type;
    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(1.0f / norm_type);

    const auto input_address = tensor_args.tmp_pow_sum.buffer()->address();
    const auto output_address = total_norm.buffer()->address();

    {
        auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, single_core);
        runtime_args[0] = input_address;
        runtime_args[3] = *reinterpret_cast<uint32_t*>(&decimal);
    }

    {
        auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, single_core);
        runtime_args[0] = output_address;
    }

    {
        auto& runtime_args = GetRuntimeArgs(program, compute_kernel_id, single_core);
        runtime_args[1] = p;
        runtime_args[2] = static_cast<uint32_t>(p_is_negative);
    }
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step2
