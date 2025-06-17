// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
MorehGroupNormBackwardGammaBetaGradOperation::MorehGroupNormBackwardGammaBetaGradFactory::cached_program_t
MorehGroupNormBackwardGammaBetaGradOperation::MorehGroupNormBackwardGammaBetaGradFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    using namespace tt;
    using namespace tt::constants;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    auto gamma_grad = outputs[0];
    auto beta_grad = outputs[1];
    auto num_groups = operation_attributes.num_groups;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = output_grad.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_shape = output_grad.padded_shape();

    const auto n = output_grad_shape[0];
    const auto c = output_grad_shape[1];
    const auto h = output_grad_shape[2];
    const auto w = output_grad_shape[3];

    const auto origin_output_grad_shape = output_grad.logical_shape();

    const auto origin_h = origin_output_grad_shape[2];
    const auto origin_w = origin_output_grad_shape[3];

    const bool is_groupnorm = true;
    const bool is_lastdim_layernorm = false;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = num_channels;  // outer_size

    const auto batch = n;
    const auto HtWt = Ht * Wt;
    const auto num_inner_tiles = batch * HtWt;  // inner_size

    const bool gamma_grad_has_value = gamma_grad.has_value();
    const bool beta_grad_has_value = beta_grad.has_value();

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
         num_channels_per_core_group_1,
         num_channels_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_channels);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_channels_per_core_group_1: {}", num_channels_per_core_group_1);
    log_debug(LogTest, "num_channels_per_core_group_2: {}", num_channels_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;                  // output_grad(==dy)
    const uint32_t in1_t = 1;                  // input(==x)
    const uint32_t in2_t = 1;                  // mean
    const uint32_t in3_t = 1;                  // rstd
    const uint32_t in4_t = 1;                  // one
    const uint32_t in5_t = do_mask_h ? 1 : 0;  // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;  // mask_w

    const uint32_t out0_t = gamma_grad_has_value ? 1 : 0;  // gamma_grad(==dgamma)
    const uint32_t out1_t = beta_grad_has_value ? 1 : 0;   // beta_grad(==dbeta)

    const uint32_t im0_t = 1;  // output(==y)
    const uint32_t im1_t = 1;  // y * dy
    const uint32_t im2_t = 1;  // Add[dy]
    const uint32_t im3_t = 1;  // Add[y * dy]
    const uint32_t im4_t = 1;  // x - mean
    const uint32_t im5_t = 1;  // dycopy

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(output_grad.dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CBIndex::c_0, in0_t},    // output_grad(==dy)
            {CBIndex::c_1, in1_t},    // input(==x)
            {CBIndex::c_2, in2_t},    // mean
            {CBIndex::c_3, in3_t},    // rstd
            {CBIndex::c_4, in4_t},    // one
            {CBIndex::c_5, in5_t},    // mask_h
            {CBIndex::c_6, in6_t},    // mask_w
            {CBIndex::c_16, out0_t},  // gamma_grad(==dgamma)
            {CBIndex::c_17, out1_t},  // beta_grad(==dbeta)
            {CBIndex::c_24, im0_t},   // output(==y)
            {CBIndex::c_25, im1_t},   // y * dy
            {CBIndex::c_26, im2_t},   // Add[dy]
            {CBIndex::c_27, im3_t},   // Add[y * dy]
            {CBIndex::c_28, im4_t},   // x - mean
            {CBIndex::c_29, im5_t},   // dycopy
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::string reader_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/gamma_beta_grad/kernels/dataflow/"
        "reader_moreh_group_norm_backward_gamma_beta_grad.cpp");
    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/gamma_beta_grad/kernels/dataflow/"
        "writer_moreh_group_norm_backward_gamma_beta_grad.cpp");

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const std::string compute_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
        "moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp");

    const std::vector<uint32_t> compute_args_group_1{
        num_channels_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        Wt,
        static_cast<uint32_t>(gamma_grad_has_value),
        static_cast<uint32_t>(beta_grad_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_groupnorm)};

    CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_channels_per_core_group_1, compute_args_group_1},
        compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_channels_per_core_group_2,
            origin_h,
            origin_w,
            num_inner_tiles,
            Wt,
            static_cast<uint32_t>(gamma_grad_has_value),
            static_cast<uint32_t>(beta_grad_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_groupnorm)};

        CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_channels_per_core_group_2, compute_args_group_2},
            compute_defines);
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

        uint32_t num_channels_per_core;
        if (core_group_1.contains(core)) {
            num_channels_per_core = num_channels_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_channels_per_core = num_channels_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            output_grad_addr,
            static_cast<uint32_t>(is_dram(output_grad)),
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            mean_addr,
            static_cast<uint32_t>(is_dram(mean)),
            rstd_addr,
            static_cast<uint32_t>(is_dram(rstd)),
            tile_offset,
            num_channels_per_core,
            num_inner_tiles,
            num_channels,
            num_groups,
            origin_h,
            origin_w,
            static_cast<uint32_t>(gamma_grad_has_value),
        };
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            gamma_grad_addr,
            static_cast<uint32_t>(gamma_grad_has_value ? is_dram(gamma_grad.value()) : 1),
            static_cast<uint32_t>(gamma_grad_has_value),
            beta_grad_addr,
            static_cast<uint32_t>(beta_grad_has_value ? is_dram(beta_grad.value()) : 1),
            static_cast<uint32_t>(beta_grad_has_value),
            tile_offset,
            num_channels_per_core,
            num_inner_tiles,
            batch,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_channels_per_core * HtWt;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void MorehGroupNormBackwardGammaBetaGradOperation::MorehGroupNormBackwardGammaBetaGradFactory::
    override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& outputs) {
    auto reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto writer_kernels_id = cached_program.shared_variables.writer_kernels_id;
    auto num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    auto output_grad_buffer = tensor_args.output_grad.buffer();
    auto input_buffer = tensor_args.input.buffer();
    auto mean_buffer = tensor_args.mean.buffer();
    auto rstd_buffer = tensor_args.rstd.buffer();

    auto gamma_grad_buffer = outputs[0]->buffer();
    auto beta_grad_buffer = outputs[1]->buffer();

    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, reader_kernels_id, core);
            runtime_args[0] = output_grad_buffer->address();
            runtime_args[2] = input_buffer->address();
            runtime_args[4] = mean_buffer->address();
            runtime_args[6] = rstd_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, writer_kernels_id, core);
            if (gamma_grad_buffer != nullptr) {
                runtime_args[0] = gamma_grad_buffer->address();
            }
            if (beta_grad_buffer != nullptr) {
                runtime_args[3] = beta_grad_buffer->address();
            }
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
