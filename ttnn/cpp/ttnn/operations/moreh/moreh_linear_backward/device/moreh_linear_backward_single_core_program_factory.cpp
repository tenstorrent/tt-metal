// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_linear_backward_device_operation.hpp"
#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/detail/util.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {

MorehBiasAddBackwardOperation::SingleCoreProgramFactory::cached_program_t
MorehBiasAddBackwardOperation::SingleCoreProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;

    auto& output_grad = tensor_args.output_grad;
    auto& bias_grad = output_tensor;

    const auto& bias_grad_shape = bias_grad.get_legacy_shape().without_padding();
    const auto& output_grad_shape_wo_padding = output_grad.get_legacy_shape().without_padding();

    auto bias_grad_memory_config = operation_attributes.bias_grad_memory_config;
    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    const bool do_mask_h = (output_grad_shape_wo_padding[-2] % constants::TILE_HEIGHT) != 0;
    const uint32_t mask_h =
        do_mask_h ? output_grad_shape_wo_padding[-2] % constants::TILE_HEIGHT : constants::TILE_HEIGHT;
    const bool do_mask_w = (output_grad_shape_wo_padding[-1] % constants::TILE_WIDTH) != 0;
    const uint32_t mask_w =
        do_mask_w ? output_grad_shape_wo_padding[-1] % constants::TILE_WIDTH : constants::TILE_WIDTH;

    const auto& output_grad_shape = output_grad.get_legacy_shape();
    uint32_t batch_num = output_grad.volume() / output_grad_shape[-2] / output_grad_shape[-1];
    uint32_t Ht = output_grad_shape[-2] / constants::TILE_HEIGHT;
    uint32_t Wt = output_grad_shape[-1] / constants::TILE_WIDTH;
    uint32_t num_tiles = output_grad.volume() / constants::TILE_HW;

    const uint32_t in0_t = 2;
    const uint32_t in1_t = 1;
    const uint32_t in2_t = (do_mask_h || do_mask_w) ? 2 : 0;  // mask_h_w

    const uint32_t out0_t = 1;
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};
    CoreCoord core = {0, 0};
    const uint32_t core_num = 1;

    Device* device = output_grad.device();
    auto arch = device->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto cb_data_format = datatype_to_dataformat_converter(output_grad.get_dtype());

    tt::operations::primary::CreateCircularBuffer(
        program,
        std::set<CoreRange>{CoreRange(core, core)},
        cb_data_format,
        {{CB::c_in0, in0_t},    // output_grad
         {CB::c_in1, in1_t},    // scaler
         {CB::c_in2, in2_t},    // mask_h_w
         {CB::c_out0, out0_t},  // bias_grad
         {CB::c_intermed0, im0_t},
         {CB::c_intermed1, im1_t, (fp32_dest_acc_en) ? tt::DataFormat::Float32 : cb_data_format}});

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(output_grad))};
    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(bias_grad))};

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/reader_moreh_bias_backward_hw.cpp";

    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/writer_moreh_bias_backward.cpp";

    const auto reader_kernel_id =
        tt::operations::primary::CreateReadKernel(program, reader_kernel_file, core, reader_compile_time_args);
    const auto writer_kernel_id =
        tt::operations::primary::CreateWriteKernel(program, writer_kernel_file, core, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    vector<uint32_t> compute_kernel_args = {};
    std::map<string, string> compute_defines;
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/moreh_bias_backward_single_core_hw.cpp";

    const auto compute_kernel_id = tt::operations::primary::CreateComputeKernel(
        program,
        compute_kernel_file,
        {core, core_num, compute_kernel_args},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {output_grad.buffer()->address(), num_tiles, 0, mask_h, mask_w, do_mask_h, do_mask_w});
    SetRuntimeArgs(program, writer_kernel_id, core, {bias_grad.buffer()->address(), 1, 0});
    SetRuntimeArgs(program, compute_kernel_id, core, {batch_num, Ht, Wt, do_mask_h, do_mask_w});

    return {std::move(program), {reader_kernel_id, writer_kernel_id}};
}

void MorehBiasAddBackwardOperation::SingleCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    auto output_grad_buffer = tensor_args.output_grad.buffer();
    auto bias_grad_buffer = tensor_return_value.buffer();
    CoreCoord core = {0, 0};
    {
        auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        runtime_args[0] = output_grad_buffer->address();
    }

    {
        auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        runtime_args[0] = bias_grad_buffer->address();
    }
}
}  // namespace ttnn::operations::moreh::moreh_linear_backward
