// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward_op.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

namespace {

operation::ProgramWithCallbacks moreh_nll_loss_backward_impl_4d(
    const Tensor &target,
    const std::optional<const Tensor> weight,
    const std::optional<const Tensor> divisor,
    const Tensor &output_grad,
    const Tensor &input_grad,
    const int32_t ignore_index,
    const bool reduction_mean,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_grad_shape = input_grad.get_legacy_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];

    auto H = input_grad_shape[-2];
    auto W = input_grad_shape[-1];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;
    auto num_inner_tile = target.volume() / N / TILE_HEIGHT / TILE_WIDTH;

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    uint32_t units_to_divide = input_grad.volume() / H / W * Ht * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(core_range, units_to_divide);

    auto arch = input_grad.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_grad.get_dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = div_up(channel_size, TILE_WIDTH);
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 1},                                                              // output_grad
            {CB::c_in1, 1, tt::DataFormat::Int32},                                       // target
            {CB::c_in2, static_cast<uint32_t>(weight_has_value ? weight_num_tile : 0)},  // weight
            {CB::c_in3, static_cast<uint32_t>(divisor_has_value ? 1 : 0)},               // divisor
            {CB::c_intermed0, 1, fp32_dest_acc_en_data_format},                          // tmp_weight
            {CB::c_intermed1, 1, fp32_dest_acc_en_data_format},                          // tmp1
            {CB::c_intermed2, 1, fp32_dest_acc_en_data_format},                          // tmp2
            {CB::c_out0, 1},                                                             // input_grad
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(target)),
        static_cast<uint32_t>(is_dram(weight)),
        static_cast<uint32_t>(is_dram(divisor)),
        static_cast<uint32_t>(is_dram(output_grad))};

    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(input_grad))};

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

    auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/kernels/"
        "reader_moreh_nll_loss_backward_4d.cpp",
        all_cores,
        reader_compile_time_args,
        reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/kernels/"
        "writer_moreh_nll_loss_backward.cpp",
        all_cores,
        writer_compile_time_args,
        writer_defines);

    const auto compute_kernel_ids = CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/kernels/"
        "moreh_nll_loss_backward_kernel.cpp",
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

    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
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
            divisor_addr,
            static_cast<uint32_t>(ignore_index),
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

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    return {
        .program = std::move(program),
        .override_runtime_arguments_callback =
            create_override_runtime_arguments_callback(reader_kernel_id, writer_kernel_id, num_cores, core_h)};
}

operation::ProgramWithCallbacks moreh_nll_loss_backward_impl_3d(
    const Tensor &target,
    const std::optional<const Tensor> weight,
    const std::optional<const Tensor> divisor,
    const Tensor &output_grad,
    const Tensor &input_grad,
    const int32_t ignore_index,
    const bool reduction_mean,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C, W)
    auto input_grad_shape = input_grad.get_legacy_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];

    auto W = input_grad_shape[-1];
    auto Ct = channel_size / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    auto target_shape = target.get_legacy_shape();
    auto num_inner_tile = target_shape[-1] / TILE_WIDTH;

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    uint32_t units_to_divide = input_grad.volume() / TILE_HEIGHT / TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(core_range, units_to_divide);

    auto arch = input_grad.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_grad.get_dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = div_up(channel_size, TILE_WIDTH);
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 1},                                                              // output_grad
            {CB::c_in1, 1, tt::DataFormat::Int32},                                       // target
            {CB::c_in2, static_cast<uint32_t>(weight_has_value ? weight_num_tile : 0)},  // weight
            {CB::c_in3, static_cast<uint32_t>(divisor_has_value ? 1 : 0)},               // divisor
            {CB::c_intermed0, 1, fp32_dest_acc_en_data_format},                          // tmp_weight
            {CB::c_intermed1, 1, fp32_dest_acc_en_data_format},                          // tmp1
            {CB::c_intermed2, 1, fp32_dest_acc_en_data_format},                          // tmp2
            {CB::c_out0, 1},                                                             // input_grad
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(target)),
        static_cast<uint32_t>(is_dram(weight)),
        static_cast<uint32_t>(is_dram(divisor)),
        static_cast<uint32_t>(is_dram(output_grad))};

    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(input_grad))};

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

    auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/kernels/"
        "reader_moreh_nll_loss_backward_3d.cpp",
        all_cores,
        reader_compile_time_args,
        reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/kernels/"
        "writer_moreh_nll_loss_backward.cpp",
        all_cores,
        writer_compile_time_args,
        writer_defines);

    const auto compute_kernel_ids = CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/kernels/"
        "moreh_nll_loss_backward_kernel.cpp",
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

    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
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
            divisor_addr,
            static_cast<uint32_t>(ignore_index),
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

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    return {
        .program = std::move(program),
        .override_runtime_arguments_callback =
            create_override_runtime_arguments_callback(reader_kernel_id, writer_kernel_id, num_cores, core_h)};
}

operation::ProgramWithCallbacks moreh_nll_loss_backward_impl_2d(
    const Tensor &target,
    const std::optional<const Tensor> weight,
    const std::optional<const Tensor> divisor,
    const Tensor &output_grad,
    const Tensor &input_grad,
    const int32_t ignore_index,
    const bool reduction_mean,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C)
    auto input_grad_shape = input_grad.get_legacy_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];

    auto W = input_grad_shape[-1];
    auto Wt = W / TILE_WIDTH;

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    uint32_t units_to_divide = input_grad.volume() / TILE_HEIGHT / TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(core_range, units_to_divide);

    auto arch = input_grad.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_grad.get_dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = div_up(channel_size, TILE_WIDTH);
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 1},                                                              // output_grad
            {CB::c_in1, 1, tt::DataFormat::Int32},                                       // target
            {CB::c_in2, static_cast<uint32_t>(weight_has_value ? weight_num_tile : 0)},  // weight
            {CB::c_in3, static_cast<uint32_t>(divisor_has_value ? 1 : 0)},               // divisor
            {CB::c_intermed0, 1, fp32_dest_acc_en_data_format},                          // tmp_weight
            {CB::c_intermed1, 1, fp32_dest_acc_en_data_format},                          // tmp1
            {CB::c_intermed2, 1, fp32_dest_acc_en_data_format},                          // tmp2
            {CB::c_out0, 1},                                                             // input_grad
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(target)),
        static_cast<uint32_t>(is_dram(weight)),
        static_cast<uint32_t>(is_dram(divisor)),
        static_cast<uint32_t>(is_dram(output_grad))};

    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(input_grad))};

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

    auto reader_kernel_id = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/kernels/"
        "reader_moreh_nll_loss_backward_2d.cpp",
        all_cores,
        reader_compile_time_args,
        reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/kernels/"
        "writer_moreh_nll_loss_backward.cpp",
        all_cores,
        writer_compile_time_args,
        writer_defines);

    const auto compute_kernel_ids = CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/kernels/"
        "moreh_nll_loss_backward_kernel.cpp",
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

    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
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
            divisor_addr,
            static_cast<uint32_t>(ignore_index),
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

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    return {
        .program = std::move(program),
        .override_runtime_arguments_callback =
            create_override_runtime_arguments_callback(reader_kernel_id, writer_kernel_id, num_cores, core_h)};
}

}  // namespace

operation::ProgramWithCallbacks moreh_nll_loss_backward_impl(
    const Tensor &target,
    const std::optional<const Tensor> weight,
    const std::optional<const Tensor> divisor,
    const Tensor &output_grad,
    const Tensor &input_grad,
    const int32_t ignore_index,
    const bool reduction_mean,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_grad_shape = input_grad.get_legacy_shape();
    auto input_grad_rank = input_grad_shape.rank();

    if (input_grad_rank == 2) {
        return moreh_nll_loss_backward_impl_2d(
            target,
            weight,
            divisor,
            output_grad,
            input_grad,
            ignore_index,
            reduction_mean,
            core_range,
            compute_kernel_config);
    }

    if (input_grad_rank == 3) {
        return moreh_nll_loss_backward_impl_3d(
            target,
            weight,
            divisor,
            output_grad,
            input_grad,
            ignore_index,
            reduction_mean,
            core_range,
            compute_kernel_config);
    }

    if (input_grad_rank >= 4) {
        return moreh_nll_loss_backward_impl_4d(
            target,
            weight,
            divisor,
            output_grad,
            input_grad,
            ignore_index,
            reduction_mean,
            core_range,
            compute_kernel_config);
    }

    return moreh_nll_loss_backward_impl_4d(
        target,
        weight,
        divisor,
        output_grad,
        input_grad,
        ignore_index,
        reduction_mean,
        core_range,
        compute_kernel_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
