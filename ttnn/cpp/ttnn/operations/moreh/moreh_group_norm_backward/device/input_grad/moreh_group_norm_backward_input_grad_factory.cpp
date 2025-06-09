// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "moreh_group_norm_backward_input_grad_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
MorehGroupNormBackwardInputGradOperation::MorehGroupNormBackwardInputGradFactory::cached_program_t
MorehGroupNormBackwardInputGradOperation::MorehGroupNormBackwardInputGradFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    using namespace tt;
    using namespace tt::constants;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    auto input_grad = outputs;
    auto gamma = tensor_args.gamma;
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

    const bool is_lastdim_layernorm = false;
    const bool is_groupnorm = true;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n * num_groups;
    const auto num_inner_tiles = (num_channels / num_groups) * Ht * Wt;

    const bool gamma_has_value = gamma.has_value();

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
         num_rows_per_core_group_1,
         num_rows_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_rows);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_rows_per_core_group_1: {}", num_rows_per_core_group_1);
    log_debug(LogTest, "num_rows_per_core_group_2: {}", num_rows_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t{1};
    const uint32_t in1_t{1};
    const uint32_t in2_t{1};
    const uint32_t in3_t{1};
    const uint32_t in4_t{1};
    const uint32_t in5_t{2};
    const uint32_t in6_t = gamma_has_value ? 1 : 0;
    const uint32_t in7_t = (do_mask_h || do_mask_w) ? 2 : 0;

    const uint32_t out0_t{1};

    uint32_t im0_t{num_inner_tiles};
    uint32_t im1_t{num_inner_tiles};
    const uint32_t im2_t{1};
    const uint32_t im3_t{1};
    const uint32_t im4_t{1};
    const uint32_t im5_t{1};
    const uint32_t im6_t{1};
    uint32_t im7_t{1};

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(output_grad.dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    const auto cb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + in7_t + out0_t + im0_t + im1_t +
                           im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) *
                          single_tile_size;
    const auto available_L1 = device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large moreh_layer_norm_backward_input_grad algorithm is selected.");
        im0_t = 1;
        im1_t = 1;
        im7_t = 0;
    } else {
        log_info(LogTest, "Small moreh_layer_norm_backward_input_grad algorithm is selected.");
    }

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CBIndex::c_0, in0_t},  // output_grad
            {CBIndex::c_1, in1_t},  // input
            {CBIndex::c_2, in2_t},  // mean
            {CBIndex::c_3, in3_t},  // rstd
            {CBIndex::c_4, in4_t},  // one
            {CBIndex::c_5, in5_t},  // inner_size(==n)
            {CBIndex::c_6, in6_t},
            {CBIndex::c_7, in7_t},
            {CBIndex::c_16, out0_t},  // input_grad
            {CBIndex::c_24, im0_t},
            {CBIndex::c_25, im1_t},
            {CBIndex::c_26, im2_t},
            {CBIndex::c_27, im3_t},
            {CBIndex::c_28, im4_t},
            {CBIndex::c_29, im5_t},
            {CBIndex::c_30, im6_t},
            {CBIndex::c_31, im7_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/kernels/dataflow/"
              "reader_moreh_group_norm_backward_input_grad_large.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/kernels/dataflow/"
              "reader_moreh_group_norm_backward_input_grad_small.cpp";

    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/kernels/dataflow/"
        "writer_moreh_group_norm_backward_input_grad.cpp");

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const auto compute_kernel_file = use_large_algorithm
                                         ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                                           "moreh_layer_norm_backward_input_grad_large_kernel.cpp"
                                         : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                                           "moreh_layer_norm_backward_input_grad_small_kernel.cpp";

    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_groupnorm)};

    CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, compute_args_group_1}, compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,
            origin_h,
            origin_w,
            num_inner_tiles,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_groupnorm)};

        CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_rows_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_addr = input.buffer()->address();
    const auto mean_addr = mean.buffer()->address();
    const auto rstd_addr = rstd.buffer()->address();

    const auto input_grad_addr = input_grad.buffer()->address();

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
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
            gamma_addr,
            static_cast<uint32_t>(gamma_has_value ? is_dram(gamma) : 1),
            static_cast<uint32_t>(gamma_has_value),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
            num_channels,
            num_groups,
            origin_h,
            origin_w,
        };
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            input_grad_addr,
            static_cast<uint32_t>(is_dram(input_grad)),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_rows_per_core * num_inner_tiles;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void MorehGroupNormBackwardInputGradOperation::MorehGroupNormBackwardInputGradFactory::override_runtime_arguments(
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
    auto gamma_buffer = tensor_args.gamma.has_value() ? tensor_args.gamma.value().buffer() : nullptr;
    auto input_grad_buffer = outputs.buffer();

    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, reader_kernels_id, core);
            runtime_args[0] = output_grad_buffer->address();
            runtime_args[2] = input_buffer->address();
            runtime_args[4] = mean_buffer->address();
            runtime_args[6] = rstd_buffer->address();
            if (gamma_buffer != nullptr) {
                runtime_args[8] = gamma_buffer->address();
            }
        }

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, writer_kernels_id, core);
            runtime_args[0] = input_grad_buffer->address();
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
