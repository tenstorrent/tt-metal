// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {

MorehSoftmaxBackwardOperation::MorehSoftmaxBackwardCLargeFactory::cached_program_t MorehSoftmaxBackwardOperation::MorehSoftmaxBackwardCLargeFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    log_info(tt::LogTest, "Large tensor algorithm selected");
    const Tensor &output = tensor_args.output_tensor;
    const Tensor &output_grad = tensor_args.output_grad_tensor;
    const Tensor &input_grad = tensor_return_value;
    uint32_t dim = operation_attributes.dim;
    const MorehSoftmaxBackwardOp op = operation_attributes.op;
    auto device = output_grad.device();
    const DeviceComputeKernelConfig compute_kernel_config =
        init_device_compute_kernel_config(device->arch(), operation_attributes.compute_kernel_config, MathFidelity::HiFi4);
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    // split work
    auto shape = input_grad.get_legacy_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    uint32_t num_tiles = input_grad.volume() / shape[dim] / H / W * Ht * Wt;

    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::operations::primary::split_work_to_cores(core_range, num_tiles);

    auto arch = input_grad.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.get_dtype());

    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CB::c_in0, 2},        // y
            {tt::CB::c_in1, 2},        // dy
            {tt::CB::c_out0, 2},       // dx
            {tt::CB::c_intermed0, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  // y * dy
            {tt::CB::c_intermed1, 2, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  // sum(y * dy)
            {tt::CB::c_intermed2, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  // dy - sum
        });

    // create read/wrtie kernel
    bool y_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dy_is_dram = output_grad.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dx_is_dram = input_grad.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    std::map<string, string> compute_defines;
    if (op == MorehSoftmaxBackwardOp::SOFTMAX) compute_defines["SOFTMAX"] = "1";
    else compute_defines["SOFTMIN"] = "1";

    if (op == MorehSoftmaxBackwardOp::LOGSOFTMAX) {
        compute_defines["LOG"] = 1;
        reader_defines["LOG"] = 1;
    }

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    auto reader_kernel_id = tt::operations::primary::CreateReadKernel(
        program, "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/reader_moreh_softmax_backward_c.cpp", all_cores, {y_is_dram, dy_is_dram}, reader_defines);
    auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(
        program, "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/writer_moreh_softmax_backward_c.cpp", all_cores, {dx_is_dram}, writer_defines);

    auto outer_stride = Ht * Wt;
    for(int i = dim ; i < shape.rank() - 2; i++ ) {
        outer_stride *= shape[i];
    }
    auto dim_size = shape[dim];
    auto inner_size = outer_stride / dim_size;

    // create compute kernel
    tt::operations::primary::CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_c_large.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, dim_size}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, dim_size}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    // Set Runtime Args
    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        vector<uint32_t> reader_args = {
            output.buffer()->address(),
            output_grad.buffer()->address(),
            num_tiles_per_core, tile_offset,
            outer_stride, inner_size,
            dim_size};

        vector<uint32_t> writer_args = {input_grad.buffer()->address(), num_tiles_per_core, tile_offset,
            outer_stride, inner_size,
            dim_size};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    return {std::move(program),
        {reader_kernel_id, writer_kernel_id, num_cores, core_h}};
}

void MorehSoftmaxBackwardOperation::MorehSoftmaxBackwardCLargeFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    auto output_tensor_buffer = tensor_args.output_tensor.buffer();
    auto output_grad_tensor_buffer = tensor_args.output_grad_tensor.buffer();

    auto input_grad_tensor_buffer = tensor_return_value.buffer();

    for (uint32_t i = 0;i < num_cores;i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = output_tensor_buffer->address();
            runtime_args[1] = output_grad_tensor_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = input_grad_tensor_buffer->address();
        }
    }
}

}
