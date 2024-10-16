// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <optional>

#include "common/constants.hpp"
#include "moreh_nll_loss_step2_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_nll_loss_step2 {

MorehNllLossStep2DeviceOperation::Factory::cached_program_t moreh_nll_loss_step2_impl_2d(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output,
    const std::string reduction,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_shape = input.get_shape().value;
    auto rank = input_shape.rank();

    auto N = input_shape[0];

    // copy 32 Btyes per core
    uint32_t units_to_divide = N / tt::constants::TILE_HEIGHT;
    const auto input_shape_without_padding = input_shape.without_padding();
    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::Device* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 1},                                                 // input
            {CB::c_in1, 1, tt::DataFormat::Int32},                          // target
            {CB::c_in2, static_cast<uint32_t>(weight_has_value ? 1 : 0)},   // weight
            {CB::c_in3, static_cast<uint32_t>(divisor_has_value ? 1 : 0)},  // divisor
            {CB::c_intermed0, 1, fp32_dest_acc_en_data_format},             // tmp_weight to reduce
            {CB::c_intermed1, 1, fp32_dest_acc_en_data_format},             // tmp_input to reduce
            {CB::c_intermed2, 1, fp32_dest_acc_en_data_format},             // tmp1
            {CB::c_intermed3, 1, fp32_dest_acc_en_data_format},             // tmp2
            {CB::c_intermed4, 1, fp32_dest_acc_en_data_format},             // tmp3
            {CB::c_out0, 1},                                                // output
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(input)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(target)),
        static_cast<uint32_t>(weight.has_value() ? tt::operations::primary::is_dram(weight.value()) : false),
        static_cast<uint32_t>(divisor.has_value() ? tt::operations::primary::is_dram(divisor.value()) : false),
    };

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(output))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;
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

    auto reader_kernel_id = tt::operations::primary::CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "reader_moreh_nll_loss_step2_2d.cpp",
        all_cores,
        reader_compile_time_args,
        reader_defines);
    auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "writer_moreh_nll_loss_step2_2d.cpp",
        all_cores,
        writer_compile_time_args,
        writer_defines);

    const auto compute_kernel_ids = tt::operations::primary::CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "moreh_nll_loss_step2_kernel.cpp",
        {
            {core_group_1, units_per_core_group_1, {units_per_core_group_1}},
            {core_group_2, units_per_core_group_2, {units_per_core_group_2}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    const auto input_addr = input.buffer()->address();
    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto divisor_addr = divisor_has_value ? divisor.value().buffer()->address() : 0;
    const auto output_addr = output.buffer()->address();

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

        vector<uint32_t> reader_args = {
            input_addr,
            target_addr,
            weight_addr,
            divisor_addr,
            static_cast<uint32_t>(ignore_index),
            units_per_core,
            tile_offset,
            origin_N,
            origin_C,
            input.element_size(),
        };

        vector<uint32_t> writer_args = {
            output_addr,
            units_per_core,
            tile_offset,
            origin_N,
        };

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{units_per_core};

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_FATAL(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    return {std::move(program),
            {.unary_reader_kernel_id = reader_kernel_id,
             .unary_writer_kernel_id = writer_kernel_id,
             .num_cores = num_cores,
             .num_cores_y = core_h}};
}

MorehNllLossStep2DeviceOperation::Factory::cached_program_t moreh_nll_loss_step2_impl_3d(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output,
    const std::string reduction,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    // split work
    auto input_shape = input.get_shape().value;
    auto rank = input_shape.rank();
    auto N = input_shape[0];

    const auto input_shape_without_padding = input_shape.without_padding();
    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];
    const auto origin_W = input_shape_without_padding[2];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::Device* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    // copy FACE_WIDTH per core
    uint32_t units_to_divide = origin_N * div_up(origin_W, tt::constants::FACE_WIDTH);

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 1},                                                 // input
            {CB::c_in1, 1, tt::DataFormat::Int32},                          // target
            {CB::c_in2, static_cast<uint32_t>(weight_has_value ? 1 : 0)},   // weight
            {CB::c_in3, static_cast<uint32_t>(divisor_has_value ? 1 : 0)},  // divisor
            {CB::c_intermed0, 1, fp32_dest_acc_en_data_format},             // tmp_weight to reduce
            {CB::c_intermed1, 1, fp32_dest_acc_en_data_format},             // tmp_input to reduce
            {CB::c_intermed2, 1, fp32_dest_acc_en_data_format},             // tmp1
            {CB::c_intermed3, 1, fp32_dest_acc_en_data_format},             // tmp2
            {CB::c_intermed4, 1, fp32_dest_acc_en_data_format},             // tmp3
            {CB::c_out0, 1},                                                // output
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(input)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(target)),
        static_cast<uint32_t>(weight.has_value() ? tt::operations::primary::is_dram(weight.value()) : false),
        static_cast<uint32_t>(divisor.has_value() ? tt::operations::primary::is_dram(divisor.value()) : false),
    };

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(output))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;
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

    auto reader_kernel_id = tt::operations::primary::CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "reader_moreh_nll_loss_step2_3d.cpp",
        all_cores,
        reader_compile_time_args,
        reader_defines);
    auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "writer_moreh_nll_loss_step2_3d.cpp",
        all_cores,
        writer_compile_time_args,
        writer_defines);

    const auto compute_kernel_ids = tt::operations::primary::CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "moreh_nll_loss_step2_kernel.cpp",
        {
            {core_group_1, units_per_core_group_1, {units_per_core_group_1}},
            {core_group_2, units_per_core_group_2, {units_per_core_group_2}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    const auto input_addr = input.buffer()->address();
    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto divisor_addr = divisor_has_value ? divisor.value().buffer()->address() : 0;
    const auto output_addr = output.buffer()->address();

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

        vector<uint32_t> reader_args = {
            input_addr,
            target_addr,
            weight_addr,
            divisor_addr,
            static_cast<uint32_t>(ignore_index),
            units_per_core,
            tile_offset,
            origin_N,
            origin_C,
            origin_W,
            input.element_size(),
        };

        vector<uint32_t> writer_args = {
            output_addr,
            units_per_core,
            tile_offset,
            origin_W,
            output.element_size(),
        };

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{units_per_core};

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_FATAL(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    return {std::move(program),
            {.unary_reader_kernel_id = reader_kernel_id,
             .unary_writer_kernel_id = writer_kernel_id,
             .num_cores = num_cores,
             .num_cores_y = core_h}};
}

MorehNllLossStep2DeviceOperation::Factory::cached_program_t moreh_nll_loss_step2_impl_4d(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output,
    const std::string reduction,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_shape = input.get_shape().value;
    auto target_shape = target.get_shape().value;
    auto rank = input_shape.rank();
    auto N = input_shape[0];
    auto channel_size = input_shape[1];

    auto H = target_shape[-2];
    auto W = target_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;
    auto num_inner_tile = target.volume() / N / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    const auto input_shape_without_padding = input_shape.without_padding();
    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::Device* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    // copy TILE per loop
    uint32_t units_to_divide = target.volume() / H / W * Ht * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = div_up(channel_size, tt::constants::TILE_WIDTH);
    tt::operations::primary::CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 1},                                                              // input
            {CB::c_in1, 1, tt::DataFormat::Int32},                                       // target
            {CB::c_in2, static_cast<uint32_t>(weight_has_value ? weight_num_tile : 0)},  // weight
            {CB::c_in3, static_cast<uint32_t>(divisor_has_value ? 1 : 0)},               // divisor
            {CB::c_intermed0, 1, fp32_dest_acc_en_data_format},                          // tmp_weight to reduce
            {CB::c_intermed1, 1, fp32_dest_acc_en_data_format},                          // tmp_input to reduce
            {CB::c_intermed2, 1, fp32_dest_acc_en_data_format},                          // tmp1
            {CB::c_intermed3, 1, fp32_dest_acc_en_data_format},                          // tmp2
            {CB::c_intermed4, 1, fp32_dest_acc_en_data_format},                          // tmp3
            {CB::c_out0, 1},                                                             // output
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(input)),
        static_cast<uint32_t>(tt::operations::primary::is_dram(target)),
        static_cast<uint32_t>(weight.has_value() ? tt::operations::primary::is_dram(weight.value()) : false),
        static_cast<uint32_t>(divisor.has_value() ? tt::operations::primary::is_dram(divisor.value()) : false),
    };

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(tt::operations::primary::is_dram(output))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;
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

    auto reader_kernel_id = tt::operations::primary::CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "reader_moreh_nll_loss_step2_4d.cpp",
        all_cores,
        reader_compile_time_args,
        reader_defines);
    auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "writer_moreh_nll_loss_step2_4d.cpp",
        all_cores,
        writer_compile_time_args,
        writer_defines);

    const auto compute_kernel_ids = tt::operations::primary::CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "moreh_nll_loss_step2_kernel.cpp",
        {
            {core_group_1, units_per_core_group_1, {units_per_core_group_1}},
            {core_group_2, units_per_core_group_2, {units_per_core_group_2}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    const auto input_addr = input.buffer()->address();
    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto divisor_addr = divisor_has_value ? divisor.value().buffer()->address() : 0;
    const auto output_addr = output.buffer()->address();

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

        vector<uint32_t> reader_args = {
            input_addr,
            target_addr,
            weight_addr,
            divisor_addr,
            static_cast<uint32_t>(ignore_index),
            units_per_core,
            tile_offset,
            origin_N,
            origin_C,
            Wt,
            num_inner_tile,
            weight_num_tile,
            input.element_size(),
        };

        vector<uint32_t> writer_args = {
            output_addr,
            units_per_core,
            tile_offset,
        };

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{units_per_core};

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_FATAL(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    return {std::move(program),
            {.unary_reader_kernel_id = reader_kernel_id,
             .unary_writer_kernel_id = writer_kernel_id,
             .num_cores = num_cores,
             .num_cores_y = core_h}};
}

MorehNllLossStep2DeviceOperation::Factory::cached_program_t MorehNllLossStep2DeviceOperation::Factory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input_tensor;
    const Tensor& target = tensor_args.target_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const std::optional<Tensor>& divisor = tensor_args.divisor_tensor;
    const Tensor& output = tensor_return_value;
    const std::string reduction = operation_attributes.reduction;
    const uint32_t ignore_index = operation_attributes.ignore_index;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    // split work
    auto input_shape = input.get_shape().value;
    auto rank = input_shape.rank();

    if (rank == 2) {
        return moreh_nll_loss_step2_impl_2d(
            input, target, weight, divisor, output, reduction, ignore_index, compute_kernel_config);
    } else if (rank == 3) {
        return moreh_nll_loss_step2_impl_3d(
            input, target, weight, divisor, output, reduction, ignore_index, compute_kernel_config);
    }

    return moreh_nll_loss_step2_impl_4d(
        input, target, weight, divisor, output, reduction, ignore_index, compute_kernel_config);
}

void MorehNllLossStep2DeviceOperation::Factory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t target_addr = tensor_args.target_tensor.buffer()->address();
    const uint32_t weight_addr =
        tensor_args.weight_tensor.has_value() ? tensor_args.weight_tensor.value().buffer()->address() : 0;
    const uint32_t divisor_addr =
        tensor_args.divisor_tensor.has_value() ? tensor_args.divisor_tensor.value().buffer()->address() : 0;
    const uint32_t ignore_index = operation_attributes.ignore_index;

    const uint32_t output_addr = tensor_return_value.buffer()->address();

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = input_addr;
            runtime_args[1] = target_addr;
            runtime_args[2] = weight_addr;
            runtime_args[3] = divisor_addr;
            runtime_args[4] = ignore_index;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step2
