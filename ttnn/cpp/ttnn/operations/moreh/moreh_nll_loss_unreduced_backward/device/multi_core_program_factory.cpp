// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "common/constants.hpp"
#include "moreh_nll_loss_unreduced_backward_device_operation.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward {

MorehNllLossUnreducedBackwardDeviceOperation::Factory::cached_program_t moreh_nll_loss_unreduced_backward_impl_2d(
    const Tensor& target,
    const std::optional<const Tensor> weight,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const int32_t ignore_index,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    // split work

    // input_grad: (N, C)
    auto input_grad_shape = input_grad.get_legacy_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];

    auto W = input_grad_shape[-1];
    auto Wt = W / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();

    tt::tt_metal::Device* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto compute_kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config_val);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.get_dtype());

    auto Ct = tt::div_up(channel_size, tt::constants::TILE_WIDTH);
    auto Nt = tt::div_up(N, tt::constants::TILE_WIDTH);
    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CB::c_in0, 1, tt::DataFormat::Int32},                          // target
            {tt::CB::c_in1, Nt},                                                // output_grad
            {tt::CB::c_in2, static_cast<uint32_t>(weight_has_value ? Ct : 0)},  // weight
            {tt::CB::c_out0, 1},                                                // input_grad
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(target)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(output_grad)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(weight))};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(input_grad))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    if (weight_has_value) {
        reader_defines["WEIGHT"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
    }

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "reader_moreh_nll_loss_unreduced_backward_2d.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "writer_moreh_nll_loss_unreduced_backward.cpp";

    auto reader_kernel_id = tt::operations::primary::CreateReadKernel(
        program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(
        program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_grad_addr = input_grad.buffer()->address();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_args = {
            target_addr,
            output_grad_addr,
            weight_addr,
            static_cast<uint32_t>(ignore_index),
            units_per_core,
            tile_offset,
            Nt,
            channel_size,
            Ct,
        };

        std::vector<uint32_t> writer_args = {input_grad_addr, units_per_core, tile_offset};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += units_per_core;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = core_h}};
}

MorehNllLossUnreducedBackwardDeviceOperation::Factory::cached_program_t moreh_nll_loss_unreduced_backward_impl_3d(
    const Tensor& target,
    const std::optional<const Tensor> weight,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const int32_t ignore_index,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    // split work

    // input_grad: (N, C, W)
    auto input_grad_shape = input_grad.get_legacy_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];

    auto W = input_grad_shape[-1];
    auto Ct = channel_size / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    auto target_shape = target.get_legacy_shape();
    auto num_inner_tile = target_shape[-1] / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();

    tt::tt_metal::Device* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto compute_kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config_val);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.get_dtype());

    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CB::c_in0, 1, tt::DataFormat::Int32},                          // target
            {tt::CB::c_in1, 1},                                                 // output_grad
            {tt::CB::c_in2, static_cast<uint32_t>(weight_has_value ? Ct : 0)},  // weight
            {tt::CB::c_out0, 1},                                                // input_grad
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(target)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(output_grad)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(weight))};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(input_grad))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    if (weight_has_value) {
        reader_defines["WEIGHT"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
    }

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "reader_moreh_nll_loss_unreduced_backward_3d.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "writer_moreh_nll_loss_unreduced_backward.cpp";

    auto reader_kernel_id = tt::operations::primary::CreateReadKernel(
        program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(
        program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    const auto target_addr = target.buffer()->address();
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto input_grad_addr = input_grad.buffer()->address();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_args = {
            target_addr,
            output_grad_addr,
            weight_addr,
            static_cast<uint32_t>(ignore_index),
            units_per_core,
            tile_offset,
            channel_size,
            Ct,
            Wt,
        };

        std::vector<uint32_t> writer_args = {input_grad_addr, units_per_core, tile_offset};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += units_per_core;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = core_h}};
}

MorehNllLossUnreducedBackwardDeviceOperation::Factory::cached_program_t moreh_nll_loss_unreduced_backward_impl_4d(
    const Tensor& target,
    const std::optional<const Tensor> weight,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const int32_t ignore_index,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    // split work
    auto input_grad_shape = input_grad.get_legacy_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];

    auto Ct = tt::div_up(channel_size, tt::constants::TILE_WIDTH);

    auto H = input_grad_shape[-2];
    auto W = input_grad_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;
    auto num_inner_tile = target.volume() / N / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();

    tt::tt_metal::Device* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.volume() / H / W * Ht * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto compute_kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config_val);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.get_dtype());

    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CB::c_in0, 1, tt::DataFormat::Int32},                          // target
            {tt::CB::c_in1, 1},                                                 // output_grad
            {tt::CB::c_in2, static_cast<uint32_t>(weight_has_value ? Ct : 0)},  // weight
            {tt::CB::c_out0, 1},                                                // input_grad
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(target)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(output_grad)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(weight))};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(input_grad))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    if (weight_has_value) {
        reader_defines["WEIGHT"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
    }

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "reader_moreh_nll_loss_unreduced_backward_4d.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "writer_moreh_nll_loss_unreduced_backward.cpp";

    auto reader_kernel_id = tt::operations::primary::CreateReadKernel(
        program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(
        program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    const auto target_addr = target.buffer()->address();
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto input_grad_addr = input_grad.buffer()->address();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_args = {
            target_addr,
            output_grad_addr,
            weight_addr,
            static_cast<uint32_t>(ignore_index),
            units_per_core,
            tile_offset,
            num_inner_tile,
            channel_size,
            Ct,
        };

        std::vector<uint32_t> writer_args = {input_grad_addr, units_per_core, tile_offset};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += units_per_core;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = core_h}};
}

MorehNllLossUnreducedBackwardDeviceOperation::Factory::cached_program_t
MorehNllLossUnreducedBackwardDeviceOperation::Factory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& target = tensor_args.target_tensor;
    const std::optional<const Tensor> weight = tensor_args.weight_tensor;
    const Tensor& output_grad = tensor_args.output_grad_tensor;

    const int32_t ignore_index = operation_attributes.ignore_index;
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = operation_attributes.compute_kernel_config;

    const Tensor& input_grad = tensor_return_value;

    // split work
    auto input_grad_shape = input_grad.get_shape();
    auto input_grad_rank = input_grad_shape.rank();

    if (input_grad_rank == 2) {
        return moreh_nll_loss_unreduced_backward_impl_2d(
            target, weight, output_grad, input_grad, ignore_index, compute_kernel_config);
    }

    if (input_grad_rank == 3) {
        return moreh_nll_loss_unreduced_backward_impl_3d(
            target, weight, output_grad, input_grad, ignore_index, compute_kernel_config);
    }

    return moreh_nll_loss_unreduced_backward_impl_4d(
        target, weight, output_grad, input_grad, ignore_index, compute_kernel_config);
}

void MorehNllLossUnreducedBackwardDeviceOperation::Factory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const uint32_t target_addr = tensor_args.target_tensor.buffer()->address();
    const uint32_t output_grad_addr = tensor_args.output_grad_tensor.buffer()->address();
    const uint32_t weight_addr =
        tensor_args.weight_tensor.has_value() ? tensor_args.weight_tensor.value().buffer()->address() : 0;
    const uint32_t ignore_index = static_cast<uint32_t>(operation_attributes.ignore_index);

    const uint32_t input_grad_addr = tensor_return_value.buffer()->address();

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = target_addr;
            runtime_args[1] = output_grad_addr;
            runtime_args[2] = weight_addr;
            runtime_args[3] = ignore_index;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = input_grad_addr;
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward
