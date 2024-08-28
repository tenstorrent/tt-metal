// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_linear_backward/moreh_linear_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_bias_backward_multi_core_h(const Tensor &output_grad, const Tensor &bias_grad, const ttnn::DeviceComputeKernelConfig &compute_kernel_config) {
    Program program{};

    DataFormat cb_data_format = datatype_to_dataformat_converter(output_grad.get_dtype());
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    Buffer *src_buffer = output_grad.buffer();
    Buffer *dst_buffer = bias_grad.buffer();
    const auto &output_grad_shape_wo_padding = output_grad.get_legacy_shape().without_padding();
    const bool do_mask_h = (output_grad_shape_wo_padding[-2] % TILE_HEIGHT) != 0;
    const uint32_t mask_h = do_mask_h ? output_grad_shape_wo_padding[-2] % TILE_HEIGHT : TILE_HEIGHT;
    const bool do_mask_w = (output_grad_shape_wo_padding[-1] % TILE_WIDTH) != 0;
    const uint32_t mask_w = do_mask_w ? output_grad_shape_wo_padding[-1] % TILE_WIDTH : TILE_WIDTH;

    const auto &output_grad_shape = output_grad.get_legacy_shape();
    uint32_t batch_num = output_grad.volume() / output_grad_shape[-2] / output_grad_shape[-1];
    uint32_t Ht = output_grad_shape[-2] / TILE_HEIGHT;
    uint32_t Wt = output_grad_shape[-1] / TILE_WIDTH;
    uint32_t num_tiles = batch_num * Ht;
    log_debug(LogOp, "{}:{} batch_num {} Ht {} Wt {} num_tiles {}", __func__, __LINE__, batch_num, Ht, Wt, num_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = output_grad.device();
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    log_debug(
        LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, Wt);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 2;
    const uint32_t in1_t = 1;
    const uint32_t in2_t = (do_mask_h || do_mask_w) ? 2 : 0;  // mask_h_w

    const uint32_t out0_t = 1;
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},    // output_grad
            {CB::c_in1, in1_t},    // scaler
            {CB::c_in2, in2_t},    // mask_h_w
            {CB::c_out0, out0_t},  // bias_grad
            {CB::c_intermed0, im0_t},
            {CB::c_intermed1, im1_t, (fp32_dest_acc_en) ? tt::DataFormat::Float32: cb_data_format}
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args{static_cast<uint32_t>(is_dram(output_grad))};
    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(bias_grad))};

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_linear_backward/kernels/reader_moreh_bias_backward_h.cpp";

    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_linear_backward/kernels/writer_moreh_bias_backward.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1};
    std::map<string, string> compute_defines;
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_linear_backward/kernels/moreh_bias_backward_multi_core_h.cpp";
    const auto compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_cols_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_cols_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        bool core_has_last_wt = (tile_offset + num_cols_per_core == Wt) ? (true) : (false);
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {src_buffer->address(),
             num_tiles,
             Wt,
             num_cols_per_core,
             tile_offset,
             mask_h,
             mask_w,
             static_cast<uint32_t>(do_mask_h),
             static_cast<uint32_t>(do_mask_w && core_has_last_wt)});

        SetRuntimeArgs(program, writer_kernel_id, core, {dst_buffer->address(), num_cols_per_core, tile_offset});

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(
                program,
                compute_kernel_1_id,
                core,
                {batch_num,
                 Ht,
                 num_cols_per_core,  // Wt_per_core
                 static_cast<uint32_t>(do_mask_h),
                 static_cast<uint32_t>(do_mask_w && core_has_last_wt)});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            TT_ASSERT(compute_kernel_2_id.has_value());
            SetRuntimeArgs(
                program,
                compute_kernel_2_id.value(),
                core,
                {batch_num,
                 Ht,
                 num_cols_per_core,  // Wt_per_core
                 static_cast<uint32_t>(do_mask_h),
                 static_cast<uint32_t>(do_mask_w && core_has_last_wt)});
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }
        tile_offset += num_cols_per_core;
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        const auto &output_grad = input_tensors.at(0);
        const auto &bias_grad = output_tensors.at(0);

        Buffer *src_buffer = output_grad.buffer();
        Buffer *dst_buffer = bias_grad.buffer();

        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}
}  // namespace primary
}  // namespace operations
}  // namespace tt
