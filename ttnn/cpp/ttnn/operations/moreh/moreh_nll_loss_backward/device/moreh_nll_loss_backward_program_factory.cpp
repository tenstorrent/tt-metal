// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>

#include <tt-metalium/constants.hpp>
#include "moreh_nll_loss_backward_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_backward {

MorehNllLossBackwardDeviceOperation::Factory::cached_program_t moreh_nll_loss_backward_impl_2d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const bool reduction_mean,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C)
    auto input_grad_shape = input_grad.padded_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];

    auto W = input_grad_shape[-1];
    auto Wt = W / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.physical_volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = tt::div_up(channel_size, tt::constants::TILE_WIDTH);
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CBIndex::c_0, 1},                                                              // output_grad
            {tt::CBIndex::c_1, 1, tt::DataFormat::Int32},                                       // target
            {tt::CBIndex::c_2, static_cast<uint32_t>(weight_has_value ? weight_num_tile : 0)},  // weight
            {tt::CBIndex::c_3, static_cast<uint32_t>(divisor_has_value ? 1 : 0)},               // divisor
            {tt::CBIndex::c_24, 1, fp32_dest_acc_en_data_format},                               // tmp_weight
            {tt::CBIndex::c_25, 1, fp32_dest_acc_en_data_format},                               // tmp1
            {tt::CBIndex::c_26, 1, fp32_dest_acc_en_data_format},                               // tmp2
            {tt::CBIndex::c_16, 1},                                                             // input_grad
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(target)),
        static_cast<uint32_t>(weight.has_value() ? is_dram(weight.value()) : false),
        static_cast<uint32_t>(divisor.has_value() ? is_dram(divisor.value()) : false),
        static_cast<uint32_t>(is_dram(output_grad))};

    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(input_grad))};

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> compute_defines{};

    if (weight_has_value) {
        reader_defines["WEIGHT"] = 1;
        compute_defines["WEIGHT"] = 1;
    }
    if (divisor_has_value) {
        reader_defines["DIVISOR"] = 1;
        compute_defines["DIVISOR"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
        compute_defines["FP32_DEST_ACC_EN"] = 1;
    }

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "reader_moreh_nll_loss_backward_2d.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "writer_moreh_nll_loss_backward.cpp";
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "moreh_nll_loss_backward_kernel.cpp";

    auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    const auto compute_kernel_ids = CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, units_per_core_group_1, {units_per_core_group_1, divisor_has_value}},
            {core_group_2, units_per_core_group_2, {units_per_core_group_2, divisor_has_value}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto divisor_addr = divisor_has_value ? divisor.value().buffer()->address() : 0;
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_grad_addr = input_grad.buffer()->address();

    // Set Runtime Args
    auto element_size = weight_has_value ? weight.value().element_size() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_args = {
            target_addr,
            output_grad_addr,
            weight_addr,
            divisor_addr,
            ignore_index,
            units_per_core,
            tile_offset,
            channel_size,
            weight_num_tile,
            element_size,
        };

        std::vector<uint32_t> writer_args = {input_grad_addr, units_per_core, tile_offset};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{units_per_core, tile_offset};

        if (core_group_1.contains(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.contains(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_FATAL(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = core_h}};
}

MorehNllLossBackwardDeviceOperation::Factory::cached_program_t moreh_nll_loss_backward_impl_3d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const bool reduction_mean,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C, W)
    auto input_grad_shape = input_grad.padded_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];

    auto W = input_grad_shape[-1];
    auto Ct = channel_size / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    auto target_shape = target.padded_shape();
    auto num_inner_tile = target_shape[-1] / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.physical_volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = tt::div_up(channel_size, tt::constants::TILE_WIDTH);
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CBIndex::c_0, 1},                                                              // output_grad
            {tt::CBIndex::c_1, 1, tt::DataFormat::Int32},                                       // target
            {tt::CBIndex::c_2, static_cast<uint32_t>(weight_has_value ? weight_num_tile : 0)},  // weight
            {tt::CBIndex::c_3, static_cast<uint32_t>(divisor_has_value ? 1 : 0)},               // divisor
            {tt::CBIndex::c_24, 1, fp32_dest_acc_en_data_format},                               // tmp_weight
            {tt::CBIndex::c_25, 1, fp32_dest_acc_en_data_format},                               // tmp1
            {tt::CBIndex::c_26, 1, fp32_dest_acc_en_data_format},                               // tmp2
            {tt::CBIndex::c_16, 1},                                                             // input_grad
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(target)),
        static_cast<uint32_t>(weight.has_value() ? is_dram(weight.value()) : false),
        static_cast<uint32_t>(divisor.has_value() ? is_dram(divisor.value()) : false),
        static_cast<uint32_t>(is_dram(output_grad))};

    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(input_grad))};

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> compute_defines{};

    if (weight_has_value) {
        reader_defines["WEIGHT"] = 1;
        compute_defines["WEIGHT"] = 1;
    }
    if (divisor_has_value) {
        reader_defines["DIVISOR"] = 1;
        compute_defines["DIVISOR"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
        compute_defines["FP32_DEST_ACC_EN"] = 1;
    }

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "reader_moreh_nll_loss_backward_3d.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "writer_moreh_nll_loss_backward.cpp";
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "moreh_nll_loss_backward_kernel.cpp";

    auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    const auto compute_kernel_ids = CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, units_per_core_group_1, {units_per_core_group_1, divisor_has_value}},
            {core_group_2, units_per_core_group_2, {units_per_core_group_2, divisor_has_value}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto divisor_addr = divisor_has_value ? divisor.value().buffer()->address() : 0;
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_grad_addr = input_grad.buffer()->address();

    // Set Runtime Args
    auto element_size = weight_has_value ? weight.value().element_size() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_args = {
            target_addr,
            output_grad_addr,
            weight_addr,
            divisor_addr,
            ignore_index,
            units_per_core,
            tile_offset,
            channel_size,
            num_inner_tile,
            weight_num_tile,
            element_size,
        };

        std::vector<uint32_t> writer_args = {input_grad_addr, units_per_core, tile_offset};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{units_per_core, tile_offset};

        if (core_group_1.contains(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.contains(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = core_h}};
}

MorehNllLossBackwardDeviceOperation::Factory::cached_program_t moreh_nll_loss_backward_impl_4d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const bool reduction_mean,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_grad_shape = input_grad.padded_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];

    auto H = input_grad_shape[-2];
    auto W = input_grad_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;
    auto num_inner_tile = target.physical_volume() / N / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.physical_volume() / H / W * Ht * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = tt::div_up(channel_size, tt::constants::TILE_WIDTH);
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CBIndex::c_0, 1},                                                              // output_grad
            {tt::CBIndex::c_1, 1, tt::DataFormat::Int32},                                       // target
            {tt::CBIndex::c_2, static_cast<uint32_t>(weight_has_value ? weight_num_tile : 0)},  // weight
            {tt::CBIndex::c_3, static_cast<uint32_t>(divisor_has_value ? 1 : 0)},               // divisor
            {tt::CBIndex::c_24, 1, fp32_dest_acc_en_data_format},                               // tmp_weight
            {tt::CBIndex::c_25, 1, fp32_dest_acc_en_data_format},                               // tmp1
            {tt::CBIndex::c_26, 1, fp32_dest_acc_en_data_format},                               // tmp2
            {tt::CBIndex::c_16, 1},                                                             // input_grad
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(target)),
        static_cast<uint32_t>(weight.has_value() ? is_dram(weight.value()) : false),
        static_cast<uint32_t>(divisor.has_value() ? is_dram(divisor.value()) : false),
        static_cast<uint32_t>(is_dram(output_grad))};

    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(input_grad))};

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> compute_defines{};

    if (weight_has_value) {
        reader_defines["WEIGHT"] = 1;
        compute_defines["WEIGHT"] = 1;
    }
    if (divisor_has_value) {
        reader_defines["DIVISOR"] = 1;
        compute_defines["DIVISOR"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
        compute_defines["FP32_DEST_ACC_EN"] = 1;
    }

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "reader_moreh_nll_loss_backward_4d.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "writer_moreh_nll_loss_backward.cpp";
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/"
        "moreh_nll_loss_backward_kernel.cpp";

    auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    const auto compute_kernel_ids = CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, units_per_core_group_1, {units_per_core_group_1, divisor_has_value}},
            {core_group_2, units_per_core_group_2, {units_per_core_group_2, divisor_has_value}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto divisor_addr = divisor_has_value ? divisor.value().buffer()->address() : 0;
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_grad_addr = input_grad.buffer()->address();

    // Set Runtime Args
    auto element_size = weight_has_value ? weight.value().element_size() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_args = {
            target_addr,
            output_grad_addr,
            weight_addr,
            divisor_addr,
            ignore_index,
            units_per_core,
            tile_offset,
            channel_size,
            num_inner_tile,
            weight_num_tile,
            element_size,
        };

        std::vector<uint32_t> writer_args = {input_grad_addr, units_per_core, tile_offset};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{units_per_core, tile_offset};

        if (core_group_1.contains(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.contains(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = core_h}};
}

MorehNllLossBackwardDeviceOperation::Factory::cached_program_t MorehNllLossBackwardDeviceOperation::Factory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& target = tensor_args.target_tensor;
    const Tensor& output_grad = tensor_args.output_grad_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const std::optional<Tensor>& divisor = tensor_args.divisor_tensor;

    const bool reduction_mean = operation_attributes.reduction_mean;
    const uint32_t ignore_index = operation_attributes.ignore_index;
    const DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config;

    const Tensor& input_grad = tensor_return_value;

    // split work
    const auto& input_grad_shape = input_grad.logical_shape();
    auto input_grad_rank = input_grad_shape.rank();

    if (input_grad_rank == 2) {
        return moreh_nll_loss_backward_impl_2d(
            target, weight, divisor, output_grad, input_grad, reduction_mean, ignore_index, compute_kernel_config);
    }

    if (input_grad_rank == 3) {
        return moreh_nll_loss_backward_impl_3d(
            target, weight, divisor, output_grad, input_grad, reduction_mean, ignore_index, compute_kernel_config);
    }

    return moreh_nll_loss_backward_impl_4d(
        target, weight, divisor, output_grad, input_grad, reduction_mean, ignore_index, compute_kernel_config);
}

void MorehNllLossBackwardDeviceOperation::Factory::override_runtime_arguments(
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
    const uint32_t divisor_addr =
        tensor_args.divisor_tensor.has_value() ? tensor_args.divisor_tensor.value().buffer()->address() : 0;
    const uint32_t ignore_index = operation_attributes.ignore_index;

    const uint32_t input_grad_addr = tensor_return_value.buffer()->address();

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = target_addr;
            runtime_args[1] = output_grad_addr;
            runtime_args[2] = weight_addr;
            runtime_args[3] = divisor_addr;
            runtime_args[4] = ignore_index;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = input_grad_addr;
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_backward
