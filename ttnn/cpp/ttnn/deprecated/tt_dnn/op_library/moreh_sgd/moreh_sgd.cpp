// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sgd/moreh_sgd_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_sgd_(
    const Tensor& param_in,
    const Tensor& grad,
    std::optional<const Tensor> momentum_buffer_in,
    const Tensor& param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const CoreRange core_range,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto shape = param_in.get_legacy_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto num = param_in.volume() / H / W;
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    bool has_momentum_buffer_out = momentum_buffer_out.has_value();

    uint32_t units_to_divide = num * Ht * Wt;
    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(core_range, units_to_divide);

    auto arch = param_in.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(param_in.get_dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 2},        // param_in
            {CB::c_in1, 2},        // grad
            {CB::c_in2, 2},        // momentum_in
            {CB::c_out0, 2},       // param_out
            {CB::c_out1, 2},       // momentum_out
            {CB::c_intermed0, 5, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  // cb_scalar_args (lr, momentum, dampening, weight_decay, one)
            {CB::c_intermed1, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  //
            {CB::c_intermed2, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  //
            {CB::c_intermed3, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  //
            {CB::c_intermed4, 1, fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format},  //
        });

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

    // create read/wrtie kernel
    auto reader_kernel_ids = CreateReadKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sgd/kernels/reader_moreh_sgd.cpp",
        all_cores,
        {
            is_dram(param_in),
            is_dram(grad),
            is_dram(momentum_buffer_in),
        },
        reader_defines);
    auto writer_kernel_ids = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sgd/kernels/writer_moreh_sgd.cpp",
        all_cores,
        {is_dram(param_out), is_dram(momentum_buffer_out)},
        writer_defines);

    // create compute kernel
    CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sgd/kernels/moreh_sgd.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2}},
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
            TT_FATAL(false, "Core not in specified core ranges");
        }

        union { float f; uint32_t u; } u_lr, u_momentum, u_dampening, u_weight_decay, u_one;
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

        SetRuntimeArgs(program, reader_kernel_ids, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_ids, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_ids = reader_kernel_ids,
                                           writer_kernel_ids = writer_kernel_ids,
                                           num_cores,
                                           core_h,
                                           has_momentum_buffer_out = has_momentum_buffer_out](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
        ) {
        TT_ASSERT(input_tensors.size() == 2);
        TT_ASSERT(optional_input_tensors.size() == 0 || optional_input_tensors.size() == 1);
        TT_ASSERT(has_momentum_buffer_out == false || output_tensors.size() == 2);

        auto param_in = input_tensors.at(0);
        auto grad = input_tensors.at(1);
        auto momentum_buffer_in = optional_input_tensors.at(0);
        auto param_out = output_tensors.at(0);

        for (uint32_t core_i = 0; core_i < num_cores; core_i++) {
            CoreCoord core = {core_i / core_h, core_i % core_h};

            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_ids, core);
                runtime_args[0] = param_in.buffer()->address();
                runtime_args[1] = grad.buffer()->address();
                if (momentum_buffer_in.has_value()) {
                    runtime_args[2] = momentum_buffer_in.value().buffer()->address();
                }
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_ids, core);
                runtime_args[0] = param_out.buffer()->address();
                if (has_momentum_buffer_out) {
                    auto momentum_buffer_out = output_tensors.at(1);
                    runtime_args[1] = momentum_buffer_out.buffer()->address();
                }
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
