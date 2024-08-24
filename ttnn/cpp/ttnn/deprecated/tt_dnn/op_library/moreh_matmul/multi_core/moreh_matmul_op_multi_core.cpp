// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_numpy/functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace operations {

namespace primary {

namespace {

void get_tensor_stride(std::vector<uint32_t>& stride,
    std::vector<uint32_t>& dim) {
    stride[0] = 1;
    for (auto i = 1; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        stride[i] = stride[i - 1] * dim[i - 1];
    }

    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(LogOp, "stride[{}] = {}", i, stride[i]);
    }
}

void get_not_bcast(
    std::vector<uint32_t>& input_not_bcast,
    std::vector<uint32_t>& input_dim,
    std::vector<uint32_t>& other_not_bcast,
    std::vector<uint32_t>& other_dim) {

    // first 2-dims are M,K and K,N
    // TODO: refaactoring
    for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        if (input_dim[i] == other_dim[i]) {
            input_not_bcast[i] = 1;
            other_not_bcast[i] = 1;
        } else {
            if (input_dim[i] == 1) {
                input_not_bcast[i] = 0;
                other_not_bcast[i] = 1;
            } else {
                input_not_bcast[i] = 1;
                other_not_bcast[i] = 0;
            }
        }
    }

    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(LogOp, "not bcast [{}] input {} other {}", i, input_not_bcast[i], other_not_bcast[i]);
    }
}

}

operation::ProgramWithCallbacks moreh_matmul_multi_core(
    const Tensor &input,
    const Tensor &other,
    const Tensor &output,
    const std::optional<const Tensor>& bias,
    bool transpose_input,
    bool transpose_other,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config) {

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program{};
    tt_metal::Device *device {input.device()};

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::DataFormat cb_data_format {datatype_to_dataformat_converter(output.get_dtype())};
    const auto single_tile_size {detail::TileSize(cb_data_format)};
    const auto num_output_tiles {output.volume() / TILE_HW};

    // input tensor
    const auto &input_shape = input.get_legacy_shape();
    const auto &input_shape_wo_padding = input_shape.without_padding();
    const auto input_rank = input_shape.rank();
    log_debug(LogOp, "input dim");
    std::vector<uint32_t> input_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(input_dim, input_shape);

    log_debug(LogOp, "input stride");
    std::vector<uint32_t> input_stride(tt::tt_metal::MAX_NUM_DIMENSIONS);
    get_tensor_stride(input_stride, input_dim);

    // other tensor
    const auto &other_shape = other.get_legacy_shape();
    const auto &other_shape_wo_padding = other_shape.without_padding();
    const auto other_rank = other_shape.rank();
    log_debug(LogOp, "other dim");
    std::vector<uint32_t> other_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(other_dim, other_shape);

    log_debug(LogOp, "other stride");
    std::vector<uint32_t> other_stride(tt::tt_metal::MAX_NUM_DIMENSIONS);
    get_tensor_stride(other_stride, other_dim);

    log_debug(LogOp, "not bcast");
    std::vector<uint32_t> input_not_bcast(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    std::vector<uint32_t> other_not_bcast(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_not_bcast(input_not_bcast, input_dim, other_not_bcast, other_dim);

    // output tensor
    const auto &output_shape = output.get_legacy_shape();
    const auto &output_shape_wo_padding = output_shape.without_padding();
    const auto output_rank = output_shape.rank();
    log_debug(LogOp, "output dim");
    std::vector<uint32_t> output_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(output_dim, output_shape);

    log_debug(LogOp, "output stride");
    std::vector<uint32_t> output_stride(tt::tt_metal::MAX_NUM_DIMENSIONS);
    get_tensor_stride(output_stride, output_dim);

    // matrix shape
    uint32_t Kt = (transpose_input) ? (input_shape[-2] / TILE_HEIGHT) : (input_shape[-1] / TILE_WIDTH);
    uint32_t Mt = (transpose_input) ? (input_shape[-1] / TILE_WIDTH) : (input_shape[-2] / TILE_HEIGHT);
    uint32_t Nt = (transpose_other) ? (other_shape[-2] / TILE_HEIGHT) : (other_shape[-1] / TILE_WIDTH);
    log_debug(LogOp, "{}:{} Mt {} Nt {} Kt {}", __func__, __LINE__, Mt, Nt, Kt);

    // bias tensor
    bool is_scalar_bias = false;
    if (bias.has_value()) {
        const auto& bias_tensor = bias.value();
        const auto& bias_shape_wo_padding = bias_tensor.get_legacy_shape().without_padding();
        is_scalar_bias = (bias_shape_wo_padding[-1] == 1) ? (true) : (false);
        log_debug(LogOp, "{}:{} bias tensor. is_scalar_bias {}", __func__, __LINE__, is_scalar_bias);
    }

    // mask
    uint32_t input_mask_h = input_shape_wo_padding[-2] % TILE_HEIGHT;
    uint32_t input_mask_w = input_shape_wo_padding[-1] % TILE_WIDTH;
    uint32_t other_mask_h = other_shape_wo_padding[-2] % TILE_HEIGHT;
    uint32_t other_mask_w = other_shape_wo_padding[-1] % TILE_WIDTH;

    bool need_input_mask_h = (input_mask_h) ? (true) : (false);
    bool need_input_mask_w = (input_mask_w) ? (true) : (false);

    bool need_other_mask_h = (other_mask_h) ? (true) : (false);
    bool need_other_mask_w = (other_mask_w) ? (true) : (false);

    if (input_mask_h == 0) {
        input_mask_h = TILE_HEIGHT;
    }
    if (input_mask_w == 0) {
        input_mask_w = TILE_WIDTH;
    }
    if (other_mask_h == 0) {
        other_mask_h = TILE_HEIGHT;
    }
    if (other_mask_w == 0) {
        other_mask_w = TILE_WIDTH;
    }

    log_debug(LogOp, "{}:{} {} {} mask_h {} mask_w {}", __func__, __LINE__,
        need_input_mask_h, need_input_mask_w,
        input_mask_h, input_mask_w);
    log_debug(LogOp, "{}:{} {} {} mask_h {} mask_w {}", __func__, __LINE__,
        need_other_mask_h, need_other_mask_w,
        other_mask_h, other_mask_w);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    log_debug(
        LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);
    ////////////////////////////////////////////////////////////////////////////
    //                         Core Grid Configuration For Workload
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] = ttnn::split_work_to_cores(grid, num_output_tiles);

    log_debug(LogOp, "{}:{} num_output_tiles: {}", __func__, __LINE__, num_output_tiles);
    log_debug(LogOp, "{}:{} num_output_tiles_per_core_group1: {}, 2: {} ", __func__, __LINE__, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2);
    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t {2};   // input
    const uint32_t in1_t {2};   // other
    const uint32_t in2_t {3};   // mask for input
    const uint32_t in3_t {3};   // mask for other
    const uint32_t in4_t {2};   // bias
    const uint32_t im0_t {1};   // temp
    const uint32_t im1_t {2};   // transpose for input
    const uint32_t im2_t {2};   // transpose for other
    const uint32_t out0_t {2};  // output

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},
            {CB::c_in1, in1_t},
            {CB::c_in2, in2_t},
            {CB::c_in3, in3_t},
            {CB::c_in4, in4_t},
            {CB::c_intermed0, im0_t, (fp32_dest_acc_en) ? tt::DataFormat::Float32: cb_data_format},
            {CB::c_intermed1, im1_t},
            {CB::c_intermed2, im2_t},
            {CB::c_out0, out0_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<string, string> reader_defines;
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(is_dram(input)), static_cast<uint32_t>(is_dram(other)), Kt,
        static_cast<uint32_t>(transpose_input),
        static_cast<uint32_t>(transpose_other),
        input_mask_h,
        input_mask_w,
        other_mask_h,
        other_mask_w,
    };

    if (bias.has_value()) {
        reader_defines["FUSE_BIAS"] = "1";
        reader_compile_time_args.push_back(static_cast<uint32_t>(is_dram(bias)));
        reader_compile_time_args.push_back(static_cast<uint32_t>(is_scalar_bias));
        log_debug(LogOp, "{}:{} bias tensor. is bias dram {}", __func__, __LINE__, is_dram(bias));
    }


    const std::vector<uint32_t> writer_compile_time_args =
        {static_cast<uint32_t>(is_dram(output))};

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_matmul/multi_core/kernels/reader_moreh_matmul.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_matmul/multi_core/kernels/writer_moreh_matmul.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);
    log_debug(LogOp, "{}:{} DMVK is_dram(input): {}, is_dram(other): {}, is_dram(output): {}", __func__, __LINE__, is_dram(input), is_dram(other), is_dram(output));

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<string, string> compute_defines;

    const auto compute_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_matmul/multi_core/kernels/moreh_matmul.cpp";
    std::vector<uint32_t> compute_args_group_1 = {
        num_output_tiles_per_core_group_1,  // num_output_tiles
        Mt,
        Nt,
        Kt,
        static_cast<uint32_t>(transpose_input),
        static_cast<uint32_t>(transpose_other),
        input_mask_h,
        input_mask_w,
        other_mask_h,
        other_mask_w
    };

    if (bias.has_value()) {
        compute_defines["FUSE_BIAS"] = "1";
        compute_args_group_1.push_back(static_cast<uint32_t>(is_scalar_bias));
    }

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const auto compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_output_tiles_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_group_2 = {
        num_output_tiles_per_core_group_2,  // num_output_tiles
        Mt,
        Nt,
        Kt,
        static_cast<uint32_t>(transpose_input),
        static_cast<uint32_t>(transpose_other),
        input_mask_h,
        input_mask_w,
        other_mask_h,
        other_mask_w
        };

        if (bias.has_value()) {
            compute_args_group_2.push_back(static_cast<uint32_t>(is_scalar_bias));
        }

        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_output_tiles_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }
    log_debug(LogOp, "{}:{} Compute ", __func__, __LINE__, static_cast<uint32_t>(transpose_input), static_cast<uint32_t>(transpose_other));

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_output_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
            std::vector<uint32_t> compute_rt_args;
            compute_rt_args.push_back(num_tiles_written);
            compute_rt_args.insert(compute_rt_args.end(), output_stride.begin(), output_stride.end());
            tt_metal::SetRuntimeArgs(
                program,
                compute_kernel_1_id,
                core,
                compute_rt_args
            );
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            TT_FATAL(compute_kernel_2_id.has_value());
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
            std::vector<uint32_t> compute_rt_args;
            compute_rt_args.push_back(num_tiles_written);
            compute_rt_args.insert(compute_rt_args.end(), output_stride.begin(), output_stride.end());
            tt_metal::SetRuntimeArgs(
                program,
                compute_kernel_2_id.value(),
                core,
                compute_rt_args
            );
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_rt_args;
        reader_rt_args.push_back(input.buffer()->address());
        reader_rt_args.push_back(other.buffer()->address());
        reader_rt_args.push_back(num_tiles_written);
        reader_rt_args.push_back(num_output_tiles_per_core);

        // TODO: move some to compile args
        reader_rt_args.insert(reader_rt_args.end(), input_stride.begin(), input_stride.end());
        reader_rt_args.insert(reader_rt_args.end(), other_stride.begin(), other_stride.end());
        reader_rt_args.insert(reader_rt_args.end(), output_stride.begin(), output_stride.end());
        reader_rt_args.insert(reader_rt_args.end(), input_not_bcast.begin(), input_not_bcast.end());
        reader_rt_args.insert(reader_rt_args.end(), other_not_bcast.begin(), other_not_bcast.end());

        if (bias.has_value()) {
            reader_rt_args.push_back(bias.value().buffer()->address());
        }

        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            reader_rt_args
        );

        tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {output.buffer()->address(),
            num_tiles_written,
            num_output_tiles_per_core});
        num_tiles_written += num_output_tiles_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id,
                                           writer_kernel_id,
                                           num_cores,
                                           num_cores_y
                                            ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
        ) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        const auto input_address = input_tensors.at(0).buffer()->address();
        const auto other_address = input_tensors.at(1).buffer()->address();
        const auto output_address = output_tensors.at(0).buffer()->address();
        const auto bias = optional_input_tensors.at(0);

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_address;
                runtime_args[1] = other_address;

                if (bias.has_value()) {
                    const auto bias_address = bias.value().buffer()->address();
                    runtime_args[runtime_args.size() - 1] = bias_address;
                }
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_address;
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
