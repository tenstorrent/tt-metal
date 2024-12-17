// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <cmath>

inline uint32_t get_block_size(uint32_t num_tiles, uint32_t max_block_size) {
    uint32_t block_size{1};
    for (uint32_t current_block_size = max_block_size; current_block_size >= 1; current_block_size >>= 1) {
        if (num_tiles % current_block_size == 0) {
            block_size = current_block_size;
            break;
        }
    }
    return block_size;
}

namespace ttnn::operations::normalization {
BatchNormOperation::BatchNormFactory::cached_program_t BatchNormOperation::BatchNormFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    using namespace tt;
    using namespace tt::constants;

    const auto& input = tensor_args.input;
    const auto& batch_mean = tensor_args.batch_mean;
    const auto& batch_var = tensor_args.batch_var;
    auto gamma = tensor_args.gamma;
    auto beta = tensor_args.beta;
    auto mean = outputs[1];
    auto rstd = outputs[2];

    auto& output = outputs[0].value();

    auto eps = operation_attributes.eps;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.get_shape();

    const auto n = input_shape.value[0];
    const auto c = input_shape.value[1];
    const auto h = input_shape.value[2];
    const auto w = input_shape.value[3];

    const auto origin_input_shape = input_shape.value.without_padding();

    const auto origin_h = origin_input_shape[2];
    const auto origin_w = origin_input_shape[3];

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const auto mask_h = do_mask_h ? origin_h % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_w % TILE_WIDTH : TILE_WIDTH;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n;
    const auto num_inner_tiles = (num_channels)*Ht * Wt;

    const bool gamma_has_value = gamma.has_value();
    const bool beta_has_value = beta.has_value();
    const bool mean_has_value = mean.has_value();
    const bool rstd_has_value = rstd.has_value();

    constexpr uint32_t MAX_BLOCK_SIZE = 8;
    const uint32_t block_size = get_block_size(num_inner_tiles, MAX_BLOCK_SIZE);
    const uint32_t block_size_batch = get_block_size(num_channels, MAX_BLOCK_SIZE);

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
    log_debug(LogTest, "block_size: {}", block_size);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_t = num_inner_tiles;                         // input
    uint32_t in1_t = block_size_batch;                        // batch_mean
    const uint32_t in2_t = 1;                                 // epsilon
    const uint32_t in3_t = gamma_has_value ? block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 1 : 0;                 // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;                 // mask_w
    uint32_t in7_t = block_size_batch;                        // batch_var

    const uint32_t out0_t = block_size;              // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    uint32_t im1_t = num_inner_tiles;                                                 // x - E[x]
    const uint32_t im0_t = 1;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im2_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta

    const auto cb_data_format = datatype_to_dataformat_converter(input.get_dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CBIndex::c_0, in0_t},    // input
            {CBIndex::c_1, in1_t},    // batch_mean
            {CBIndex::c_2, in2_t},    // eps
            {CBIndex::c_3, in3_t},    // gamma
            {CBIndex::c_4, in4_t},    // beta
            {CBIndex::c_5, in5_t},    // mask_h
            {CBIndex::c_6, in6_t},    // mask_w
            {CBIndex::c_7, in7_t},    // batch_var
            {CBIndex::c_16, out0_t},  // output
            {CBIndex::c_17, out1_t},  // mean
            {CBIndex::c_18, out2_t},  // rstd
            {CBIndex::c_24, im0_t},   // 1.0/(sqrt(Var[x] + eps))
            {CBIndex::c_25, im1_t},   // x - E[x]
            {CBIndex::c_26, im2_t},   // y * gamm + beta
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp";

    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp");

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp";

    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        block_size,
        block_size_batch,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value)};

    CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, compute_args_group_1}, compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,
            origin_h,
            origin_w,
            num_inner_tiles,
            block_size,
            block_size_batch,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(beta_has_value),
            static_cast<uint32_t>(mean_has_value),
            static_cast<uint32_t>(rstd_has_value)};

        CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_rows_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();
    const auto batch_mean_addr = batch_mean.buffer()->address();
    const auto batch_var_addr = batch_var.buffer()->address();

    const auto output_addr = output.buffer()->address();
    const auto mean_addr = mean_has_value ? mean.value().buffer()->address() : 0;
    const auto rstd_addr = rstd_has_value ? rstd.value().buffer()->address() : 0;

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;
    const auto beta_addr = beta_has_value ? beta.value().buffer()->address() : 0;

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
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            batch_mean_addr,
            static_cast<uint32_t>(is_dram(batch_mean)),
            batch_var_addr,
            static_cast<uint32_t>(is_dram(batch_var)),
            gamma_addr,
            static_cast<uint32_t>(is_dram(gamma)),
            static_cast<uint32_t>(gamma_has_value),
            beta_addr,
            static_cast<uint32_t>(is_dram(beta)),
            static_cast<uint32_t>(beta_has_value),
            *reinterpret_cast<uint32_t*>(&eps),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
            num_channels,
            origin_h,
            origin_w,
            block_size,
            block_size_batch,
        };
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            output_addr,
            static_cast<uint32_t>(is_dram(output)),
            mean_addr,
            static_cast<uint32_t>(mean_has_value ? is_dram(mean.value()) : 1),
            static_cast<uint32_t>(mean_has_value),
            rstd_addr,
            static_cast<uint32_t>(rstd_has_value ? is_dram(rstd.value()) : 1),
            static_cast<uint32_t>(rstd_has_value),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
            block_size,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_rows_per_core * num_inner_tiles;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void BatchNormOperation::BatchNormFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto input_buffer = tensor_args.input.buffer();
    auto batch_mean_buffer = tensor_args.batch_mean.buffer();
    auto batch_var_buffer = tensor_args.batch_var.buffer();
    auto gamma_buffer = tensor_args.gamma.has_value() ? tensor_args.gamma.value().buffer() : nullptr;
    auto beta_buffer = tensor_args.beta.has_value() ? tensor_args.beta.value().buffer() : nullptr;

    auto ouput_buffer = tensor_return_value[0]->buffer();
    auto mean_buffer = tensor_return_value[1]->buffer();
    auto rstd_buffer = tensor_return_value[2]->buffer();

    auto reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto writer_kernels_id = cached_program.shared_variables.writer_kernels_id;
    auto num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, reader_kernels_id, core);
            runtime_args[0] = input_buffer->address();
            runtime_args[2] = batch_mean_buffer->address();
            runtime_args[4] = batch_var_buffer->address();
            if (gamma_buffer != nullptr) {
                runtime_args[6] = gamma_buffer->address();
            }
            if (beta_buffer != nullptr) {
                runtime_args[9] = beta_buffer->address();
            }
        }

        {
            auto& runtime_args = GetRuntimeArgs(cached_program.program, writer_kernels_id, core);
            runtime_args[0] = ouput_buffer->address();
            if (mean_buffer != nullptr) {
                runtime_args[2] = mean_buffer->address();
            }
            if (rstd_buffer != nullptr) {
                runtime_args[5] = rstd_buffer->address();
            }
        }
    }
}
}  // namespace ttnn::operations::normalization
