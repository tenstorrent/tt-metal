// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_sgd_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_sgd {
MorehSgdOperation::ProgramFactory::cached_program_t MorehSgdOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& param_in = tensor_args.param_in;
    auto& grad = tensor_args.grad;
    const std::optional<Tensor>& momentum_buffer_in = tensor_args.momentum_buffer_in;

    auto& output_tensors = output_tensor;
    auto& param_out = output_tensors.at(0).value();
    auto& momentum_buffer_out = output_tensors.at(1);

    auto lr = operation_attributes.lr;
    auto momentum = operation_attributes.momentum;
    auto dampening = operation_attributes.dampening;
    auto weight_decay = operation_attributes.weight_decay;
    auto nesterov = operation_attributes.nesterov;
    auto momentum_initialized = operation_attributes.momentum_initialized;

    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    auto shape = param_in.get_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto num = param_in.volume() / H / W;
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    bool has_momentum_buffer_out = momentum_buffer_out.has_value();

    Program program{};

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::Device* device = param_in.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t units_to_divide = num * Ht * Wt;
    uint32_t core_w = grid.x;
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, units_to_divide);

    auto arch = param_in.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(param_in.get_dtype());
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CB::c_in0, 2},   // param_in
            {tt::CB::c_in1, 2},   // grad
            {tt::CB::c_in2, 2},   // momentum_in
            {tt::CB::c_out0, 2},  // param_out
            {tt::CB::c_out1, 2},  // momentum_out

            {tt::CB::c_intermed0,
             5,
             intermed_cb_format},  // cb_scalar_args (lr, momentum, dampening, weight_decay, one)
            {tt::CB::c_intermed1, 1, intermed_cb_format},  //
            {tt::CB::c_intermed2, 1, intermed_cb_format},  //
            {tt::CB::c_intermed3, 1, intermed_cb_format},  //
            {tt::CB::c_intermed4, 1, intermed_cb_format},  //
        });

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels defines
    ////////////////////////////////////////////////////////////////////////////
    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;
    std::map<string, string> compute_defines;

    if (weight_decay != 0) {
        reader_defines["WEIGHT_DECAY"] = 1;
        compute_defines["WEIGHT_DECAY"] = 1;
    }

    if (momentum != 0) {
        reader_defines["MOMENTUM"] = 1;
        compute_defines["MOMENTUM"] = 1;
        writer_defines["MOMENTUM"] = 1;
    }

    if (momentum_initialized) {
        reader_defines["MOMENTUM_INITIALIZED"] = 1;
        compute_defines["MOMENTUM_INITIALIZED"] = 1;
    }

    if (nesterov) {
        reader_defines["NESTEROV"] = 1;
        compute_defines["NESTEROV"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(param_in)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(grad)),
        static_cast<uint32_t>(
            momentum_buffer_in.has_value() ? tt::operations::primary::is_dram(momentum_buffer_in.value()) : 0)};

    std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(tt::operations::primary::is_dram(param_out))};
    if (has_momentum_buffer_out)
        writer_compile_time_args.push_back(
            static_cast<uint32_t>(tt::operations::primary::is_dram(momentum_buffer_out.value())));

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/"
        "reader_moreh_sgd.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/"
        "writer_moreh_sgd.cpp";

    const auto reader_kernel_id = tt::operations::primary::CreateReadKernel(
        program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(
        program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_tiles_per_core_group_1};

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/"
        "moreh_sgd.cpp";

    auto compute_kernel_id = tt ::operations::primary::CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto param_in_addr = param_in.buffer()->address();
    const auto grad_addr = grad.buffer()->address();
    const auto momentum_buffer_in_addr =
        momentum_buffer_in.has_value() ? momentum_buffer_in.value().buffer()->address() : 0;

    const auto param_out_addr = param_out.buffer()->address();
    const auto momentum_buffer_out_addr =
        momentum_buffer_out.has_value() ? momentum_buffer_out->buffer()->address() : 0;

    auto core_x_offset = 0;
    auto core_y_offset = 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        union {
            float f;
            uint32_t u;
        } u_lr, u_momentum, u_dampening, u_weight_decay, u_one;
        u_lr.f = lr;
        u_momentum.f = momentum;
        u_dampening.f = dampening;
        u_weight_decay.f = weight_decay;
        u_one.f = 1.0f;

        vector<uint32_t> reader_args = {
            param_in.buffer()->address(),
            grad.buffer()->address(),
            momentum_buffer_in.has_value() ? momentum_buffer_in.value().buffer()->address() : 0,
            num_tiles_per_core,
            tile_offset,
            u_lr.u,
            u_momentum.u,
            u_dampening.u,
            u_weight_decay.u,
            u_one.u,
        };

        vector<uint32_t> writer_args = {
            param_out.buffer()->address(),
            momentum_buffer_out.has_value() ? momentum_buffer_out.value().buffer()->address() : 0,
            num_tiles_per_core,
            tile_offset,
        };

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h, has_momentum_buffer_out}};
}

void MorehSgdOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    auto param_in_buffer = tensor_args.param_in.buffer();
    auto grad_buffer = tensor_args.grad.buffer();
    auto momentum_buffer_in_buffer =
        tensor_args.momentum_buffer_in.has_value() ? tensor_args.momentum_buffer_in->buffer() : 0;

    auto param_out_buffer = tensor_return_value.at(0)->buffer();
    auto momentum_buffer_out_buffer = tensor_return_value.at(1)->buffer();

    auto num_cores = cached_program.shared_variables.num_cores;
    auto core_h = cached_program.shared_variables.core_h;
    auto has_momentum_buffer_out = cached_program.shared_variables.has_momentum_buffer_out;

    TT_ASSERT(has_momentum_buffer_out == false || tensor_return_value.size() == 2);

    for (uint32_t core_i = 0; core_i < num_cores; core_i++) {
        CoreCoord core = {core_i / core_h, core_i % core_h};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = param_in_buffer->address();
            runtime_args[1] = grad_buffer->address();
            if (tensor_args.momentum_buffer_in.has_value()) {
                runtime_args[2] = momentum_buffer_in_buffer->address();
                ;
            }
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = param_out_buffer->address();
            if (has_momentum_buffer_out) {
                runtime_args[1] = momentum_buffer_out_buffer->address();
            }
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_sgd
