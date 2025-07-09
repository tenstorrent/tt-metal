// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {

MorehSoftmaxBackwardOperation::MorehSoftmaxBackwardWSmallFactory::cached_program_t
MorehSoftmaxBackwardOperation::MorehSoftmaxBackwardWSmallFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input_grad) {
    log_info(tt::LogTest, "Small tensor algorithm selected");
    const auto& output = tensor_args.output_tensor;
    const auto& output_grad = tensor_args.output_grad_tensor;
    const auto op = operation_attributes.op;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto device = output_grad.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    // split work
    auto shape = input_grad.padded_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    auto num = input_grad.physical_volume() / H / W;

    uint32_t num_kernel_rows = num * Ht;
    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, num_kernel_rows);

    auto arch = input_grad.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CBIndex::c_0, Wt},                                                             // output
            {tt::CBIndex::c_1, Wt},                                                             // output_grad
            {tt::CBIndex::c_2, 1},                                                              // scaler
            {tt::CBIndex::c_3, 1},                                                              // mask
            {tt::CBIndex::c_16, 2},                                                             // input_grad
            {tt::CBIndex::c_24, Wt, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  // output * output_grad
            {tt::CBIndex::c_25, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},   // reduce
            {tt::CBIndex::c_26, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},   // dy - sum
        });
    // create read/wrtie kernel
    bool y_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dy_is_dram = output_grad.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dx_is_dram = input_grad.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;

    auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/reader_moreh_softmax_backward_w.cpp",
        all_cores,
        {y_is_dram, dy_is_dram},
        reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/writer_moreh_softmax_w.cpp",
        all_cores,
        {dx_is_dram},
        writer_defines);

    std::map<std::string, std::string> compute_defines;
    if (op == MorehSoftmaxBackwardOp::SOFTMAX) {
        compute_defines["SOFTMAX"] = "1";
    } else {
        compute_defines["SOFTMIN"] = "1";
    }

    if (op == MorehSoftmaxBackwardOp::LOGSOFTMAX) {
        compute_defines["LOG"] = 1;
    }

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    // create compute kernel
    CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, Wt}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, Wt}},
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
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        float scaler = 1.0f;
        uint32_t mask_w = input_grad.logical_shape()[-1] % tt::constants::TILE_WIDTH;
        if (mask_w == 0) {
            mask_w = tt::constants::TILE_WIDTH;
        }
        std::vector<uint32_t> reader_args = {
            output.buffer()->address(),
            output_grad.buffer()->address(),
            num_tiles_per_core,
            tile_offset,
            Wt,
            *reinterpret_cast<uint32_t*>(&scaler),
            mask_w};

        std::vector<uint32_t> writer_args = {input_grad.buffer()->address(), num_tiles_per_core, tile_offset, Wt};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core * Wt;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h}};
}

void MorehSoftmaxBackwardOperation::MorehSoftmaxBackwardWSmallFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input_grad) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = tensor_args.output_tensor.buffer()->address();
            runtime_args[1] = tensor_args.output_grad_tensor.buffer()->address();
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = input_grad.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_softmax_backward
