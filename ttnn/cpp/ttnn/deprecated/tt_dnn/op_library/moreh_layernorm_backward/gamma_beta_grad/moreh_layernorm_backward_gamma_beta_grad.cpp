// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_layernorm_backward_gamma_beta_grad_impl(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<const Tensor> gamma_grad,
    const std::optional<const Tensor> beta_grad) {
    using namespace tt::constants;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Device* device = output_grad.device();
    Program program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_shape = output_grad.get_legacy_shape();
    const auto output_grad_shape_without_padding = output_grad_shape.without_padding();
    const auto output_grad_rank = output_grad_shape.rank();

    const bool is_lastdim_layernorm = normalized_dims == 1;
    const bool is_groupnorm = false;

    const auto origin_H = output_grad_shape_without_padding[-2];
    const auto origin_W = output_grad_shape_without_padding[-1];

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && is_lastdim_layernorm;
    const uint32_t mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    const auto mean_rstd_shape = mean.get_legacy_shape();
    const auto mean_rstd_shape_without_padding = mean_rstd_shape.without_padding();
    auto mean_rstd_height = mean_rstd_shape_without_padding[-2];
    auto mean_rstd_width = mean_rstd_shape_without_padding[-1];

    auto num_inner = compute_inner(output_grad_shape, normalized_dims);
    auto num_outer = compute_outer(output_grad_shape, normalized_dims);

    const bool gamma_grad_has_value = gamma_grad.has_value();
    const bool beta_grad_has_value = beta_grad.has_value();
    TT_ASSERT(gamma_grad_has_value || beta_grad_has_value);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = ttnn::split_work_to_cores(grid, num_inner);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;                  // output_grad(==dy)
    const uint32_t in1_t = 1;                  // input(==x)
    const uint32_t in2_t = 1;                  // mean
    const uint32_t in3_t = 1;                  // rstd
    const uint32_t in4_t = 1;                  // scaler
    const uint32_t in5_t = do_mask_h ? 1 : 0;  // mask_h

    const uint32_t out0_t = 1;  // gamma_grad(==dgamma)
    const uint32_t out1_t = 1;  // beta_grad(==dbeta)

    const uint32_t im0_t = 1;  // output(==y)
    const uint32_t im1_t = 1;  // y * dy
    const uint32_t im2_t = 1;  // Add[dy]
    const uint32_t im3_t = 1;  // Add[y * dy]
    const uint32_t im4_t = 1;  // x - mean
    const uint32_t im5_t = 1;  // dycopy

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(output_grad.get_dtype());
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},        // output_grad(==dy)
            {CB::c_in1, in1_t},        // input(==x)
            {CB::c_in2, in2_t},        // mean
            {CB::c_in3, in3_t},        // rstd
            {CB::c_in4, in4_t},        // scaler
            {CB::c_in5, in5_t},        // mask_h
            {CB::c_out0, out0_t},      // gamma_grad(==dgamma)
            {CB::c_out1, out1_t},      // beta_grad(==dbeta)
            {CB::c_intermed0, im0_t, intermed_cb_format},  // output(==y)
            {CB::c_intermed1, im1_t, intermed_cb_format},  // y * dy
            {CB::c_intermed2, im2_t, intermed_cb_format},  // Add[dy]
            {CB::c_intermed3, im3_t, intermed_cb_format},  // Add[y * dy]
            {CB::c_intermed4, im4_t, intermed_cb_format},  // x - mean
            {CB::c_intermed5, im5_t, intermed_cb_format},  // dycopy
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(output_grad)),
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(is_dram(mean)),
        static_cast<uint32_t>(is_dram(rstd)),
        static_cast<uint32_t>(gamma_grad_has_value),
        static_cast<uint32_t>(do_mask_h)};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(is_dram(gamma_grad)),
        static_cast<uint32_t>(is_dram(beta_grad)),
        static_cast<uint32_t>(gamma_grad_has_value),
        static_cast<uint32_t>(beta_grad_has_value)};

    std::map<string, string> reader_defines{};
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";
    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = "1";
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/kernels/"
        "reader_moreh_layernorm_backward_gamma_beta_grad.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/kernels/"
        "writer_moreh_layernorm_backward_gamma_beta_grad.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    const std::vector<uint32_t> compute_args_group_1{
        num_cols_per_core_group_1,
        origin_H,
        origin_W,
        num_outer,
        num_inner,
        static_cast<uint32_t>(gamma_grad_has_value),
        static_cast<uint32_t>(beta_grad_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_groupnorm)};

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/kernels/"
        "moreh_layernorm_backward_gamma_beta_grad_kernel.cpp";

    CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_cols_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_cols_per_core_group_2,
            origin_H,
            origin_W,
            num_outer,
            num_inner,
            static_cast<uint32_t>(gamma_grad_has_value),
            static_cast<uint32_t>(beta_grad_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_groupnorm)};

        CreateComputeKernel(
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
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_addr = input.buffer()->address();
    const auto mean_addr = mean.buffer()->address();
    const auto rstd_addr = rstd.buffer()->address();

    const auto gamma_grad_addr = gamma_grad_has_value ? gamma_grad.value().buffer()->address() : 0;
    const auto beta_grad_addr = beta_grad_has_value ? beta_grad.value().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_cols_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        const std::vector<uint32_t> reader_runtime_args{
            output_grad_addr, input_addr, mean_addr, rstd_addr, num_cols_per_core, num_outer, num_inner, tile_offset, mask_h, normalized_dims, mean_rstd_height, mean_rstd_width};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{
            gamma_grad_addr, beta_grad_addr, num_cols_per_core, tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_cols_per_core;
    }

    return {
        .program = std::move(program),
        .override_runtime_arguments_callback =
            create_override_runtime_arguments_callback(reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y)};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
