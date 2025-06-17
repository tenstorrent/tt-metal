// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "moreh_adamw_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_adamw {
MorehAdamWDeviceOperation::MultiCore::cached_program_t MorehAdamWDeviceOperation::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& param_in = tensor_args.param_in;
    const Tensor& grad = tensor_args.grad;
    const Tensor& exp_avg_in = tensor_args.exp_avg_in;
    const Tensor& exp_avg_sq_in = tensor_args.exp_avg_sq_in;

    float lr = operation_attributes.lr;
    float beta1 = operation_attributes.beta1;
    float beta2 = operation_attributes.beta2;
    float eps = operation_attributes.eps;
    float weight_decay = operation_attributes.weight_decay;
    uint32_t step = operation_attributes.step;
    bool amsgrad = operation_attributes.amsgrad;

    uint32_t num_units = param_in.physical_volume() / tt::constants::TILE_HW;

    const std::optional<Tensor>& max_exp_avg_sq_in = tensor_args.max_exp_avg_sq_in;

    // It's guarantee that param_out, exp_avg_out, exp_avg_sq_out are created.
    const Tensor& param_out = tensor_return_value.at(0).value();
    const Tensor& exp_avg_out = tensor_return_value.at(1).value();
    const Tensor& exp_avg_sq_out = tensor_return_value.at(2).value();
    const std::optional<Tensor>& max_exp_avg_sq_out =
        amsgrad ? std::optional<const Tensor>{tensor_return_value.at(3)} : std::nullopt;

    DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config;

    Program program{};

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::IDevice* device = param_in.device();
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        tt_metal::split_work_to_cores(grid, num_units);

    auto arch = param_in.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto data_format = tt_metal::datatype_to_dataformat_converter(param_in.dtype());
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CBIndex::c_0, 1},                      // param_in
            {CBIndex::c_1, 1},                      // grad
            {CBIndex::c_2, 1},                      // exp_avg_in
            {CBIndex::c_3, 1},                      // exp_avg_sq_in
            {CBIndex::c_4, 1},                      // max_exp_avg_sq_in (optional)
            {CBIndex::c_5, 5, intermed_cb_format},  // lr, beta1, beta2, eps, weight_decay
            {CBIndex::c_6, 1, intermed_cb_format},  // 1.0f

            {CBIndex::c_24, 1, intermed_cb_format},  // tmp_grad
            {CBIndex::c_25, 1, intermed_cb_format},  // tmp_exp_avg
            {CBIndex::c_26, 1, intermed_cb_format},  // tmp_exp_avg_sq
            {CBIndex::c_27, 1, intermed_cb_format},  // tmp_max_exp_avg_sq
            {CBIndex::c_28, 1, intermed_cb_format},  // beta1_exponent
            {CBIndex::c_29, 1, intermed_cb_format},  // beta2_exponent
            {CBIndex::c_30, 1, intermed_cb_format},  // tmp1
            {CBIndex::c_31, 1, intermed_cb_format},  // tmp2

            {CBIndex::c_16, 1},  // param_out
            {CBIndex::c_17, 1},  // exp_avg_out
            {CBIndex::c_18, 1},  // exp_avg_sq_out
            {CBIndex::c_19, 1},  // max_exp_avg_sq_out (optional)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(param_in)),
        static_cast<uint32_t>(is_dram(grad)),
        static_cast<uint32_t>(is_dram(exp_avg_in)),
        static_cast<uint32_t>(is_dram(exp_avg_sq_in)),
        static_cast<uint32_t>(max_exp_avg_sq_in.has_value() ? is_dram(max_exp_avg_sq_in.value()) : false)};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(is_dram(param_out)),
        static_cast<uint32_t>(is_dram(exp_avg_out)),
        static_cast<uint32_t>(is_dram(exp_avg_sq_out)),
        static_cast<uint32_t>(max_exp_avg_sq_out.has_value() ? is_dram(max_exp_avg_sq_out.value()) : false)};

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/"
        "reader_moreh_adamw.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/"
        "writer_moreh_adamw.cpp";

    std::map<std::string, std::string> data_movement_defines{};
    std::map<std::string, std::string> compute_defines{};
    if (amsgrad) {
        data_movement_defines["AMSGRAD"] = "1";
        compute_defines["AMSGRAD"] = "1";
    }
    if (fp32_dest_acc_en) {
        data_movement_defines["FP32_DEST_ACC_EN"] = "1";
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, data_movement_defines);
    const auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, data_movement_defines);

    const std::vector<uint32_t> compute_args_group_1{num_units_per_core_group_1};
    const std::vector<uint32_t> compute_args_group_2{num_units_per_core_group_2};

    auto compute_kernel_ids = CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/moreh_adamw.cpp",
        {
            {core_group_1, num_units_per_core_group_1, compute_args_group_1},
            {core_group_2, num_units_per_core_group_2, compute_args_group_2},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t param_in_addr = param_in.buffer()->address();
    const uint32_t grad_addr = grad.buffer()->address();
    const uint32_t exp_avg_in_addr = exp_avg_in.buffer()->address();
    const uint32_t exp_avg_sq_in_addr = exp_avg_sq_in.buffer()->address();
    const uint32_t max_exp_avg_sq_in_addr =
        max_exp_avg_sq_in.has_value() ? max_exp_avg_sq_in.value().buffer()->address() : 0;

    const uint32_t param_out_addr = param_out.buffer()->address();
    const uint32_t exp_avg_out_addr = exp_avg_out.buffer()->address();
    const uint32_t exp_avg_sq_out_addr = exp_avg_sq_out.buffer()->address();
    const uint32_t max_exp_avg_sq_out_addr =
        max_exp_avg_sq_out.has_value() ? max_exp_avg_sq_out.value().buffer()->address() : 0;
    float beta1_exponent = std::pow(beta1, step);
    float beta2_exponent = std::pow(beta2, step);

    union {
        float f;
        uint32_t u;
    } f2u_lr, f2u_beta1, f2u_beta2, f2u_eps, f2u_weight_decay, f2u_beta1_exponent, f2u_beta2_exponent;
    f2u_lr.f = lr;
    f2u_beta1.f = beta1;
    f2u_beta2.f = beta2;
    f2u_eps.f = eps;
    f2u_weight_decay.f = weight_decay;
    f2u_beta1_exponent.f = beta1_exponent;
    f2u_beta2_exponent.f = beta2_exponent;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        const std::vector<uint32_t> reader_runtime_args{
            param_in_addr,
            grad_addr,
            exp_avg_in_addr,
            exp_avg_sq_in_addr,
            max_exp_avg_sq_in_addr,
            f2u_lr.u,
            f2u_beta1.u,
            f2u_beta2.u,
            f2u_eps.u,
            f2u_weight_decay.u,
            f2u_beta1_exponent.u,
            f2u_beta2_exponent.u,
            step,
            static_cast<uint32_t>(amsgrad),
            num_tiles_per_core,
            tile_offset};

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{
            param_out_addr,
            exp_avg_out_addr,
            exp_avg_sq_out_addr,
            max_exp_avg_sq_out_addr,
            num_tiles_per_core,
            tile_offset};
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{step};

        if (core_group_1.contains(core)) {
            tt_metal::SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.contains(core)) {
            tt_metal::SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        tile_offset += num_tiles_per_core;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .compute_kernel_group1_id = compute_kernel_ids[0],
         .compute_kernel_group2_id = compute_kernel_ids[1],
         .core_group_1 = core_group_1,
         .core_group_2 = core_group_2,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void MorehAdamWDeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& compute_kernel_1_id = cached_program.shared_variables.compute_kernel_group1_id;
    auto& compute_kernel_2_id = cached_program.shared_variables.compute_kernel_group2_id;
    auto& core_group_1 = cached_program.shared_variables.core_group_1;
    auto& core_group_2 = cached_program.shared_variables.core_group_2;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const uint32_t param_in_addr = tensor_args.param_in.buffer()->address();
    const uint32_t grad_addr = tensor_args.grad.buffer()->address();
    const uint32_t exp_avg_in_addr = tensor_args.exp_avg_in.buffer()->address();
    const uint32_t exp_avg_sq_in_addr = tensor_args.exp_avg_sq_in.buffer()->address();
    const uint32_t max_exp_avg_sq_in_addr =
        tensor_args.max_exp_avg_sq_in.has_value() ? tensor_args.max_exp_avg_sq_in.value().buffer()->address() : 0;

    const uint32_t param_out_addr =
        tensor_return_value.at(0).has_value() ? tensor_return_value.at(0).value().buffer()->address() : 0;
    const uint32_t exp_avg_out_addr =
        tensor_return_value.at(1).has_value() ? tensor_return_value.at(1).value().buffer()->address() : 0;
    const uint32_t exp_avg_sq_out_addr =
        tensor_return_value.at(2).has_value() ? tensor_return_value.at(2).value().buffer()->address() : 0;
    const uint32_t max_exp_avg_sq_out_addr =
        operation_attributes.amsgrad ? tensor_return_value.at(3).value().buffer()->address() : 0;

    union {
        float f;
        uint32_t u;
    } f2u_lr, f2u_beta1_exponent, f2u_beta2_exponent;

    f2u_lr.f = operation_attributes.lr;
    // Recalculate pow(beta, step)
    f2u_beta1_exponent.f = std::pow(operation_attributes.beta1, operation_attributes.step);
    f2u_beta2_exponent.f = std::pow(operation_attributes.beta2, operation_attributes.step);

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = param_in_addr;
            runtime_args[1] = grad_addr;
            runtime_args[2] = exp_avg_in_addr;
            runtime_args[3] = exp_avg_sq_in_addr;
            runtime_args[4] = max_exp_avg_sq_in_addr;
            runtime_args[5] = f2u_lr.u;
            runtime_args[10] = f2u_beta1_exponent.u;
            runtime_args[11] = f2u_beta2_exponent.u;
            runtime_args[12] = operation_attributes.step;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = param_out_addr;
            runtime_args[1] = exp_avg_out_addr;
            runtime_args[2] = exp_avg_sq_out_addr;
            runtime_args[3] = max_exp_avg_sq_out_addr;
        }

        {
            if (core_group_1.contains(core)) {
                auto& runtime_args = GetRuntimeArgs(program, compute_kernel_1_id, core);
                runtime_args[0] = operation_attributes.step;
            } else if (core_group_2.contains(core)) {
                auto& runtime_args = GetRuntimeArgs(program, compute_kernel_2_id, core);
                runtime_args[0] = operation_attributes.step;
            } else {
                TT_THROW("Core not in specified core ranges.");
            }
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_adamw
