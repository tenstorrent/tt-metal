// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

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

namespace ttnn::operations::moreh::moreh_group_norm {
MorehGroupNormOperation::MorehGroupNormFactory::cached_program_t MorehGroupNormOperation::MorehGroupNormFactory::create(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &outputs) {
    using namespace tt;
    using namespace tt::constants;
    using namespace tt::operations::primary;

    const auto &input = tensor_args.input;
    auto gamma = tensor_args.gamma;
    auto beta = tensor_args.beta;
    auto mean = outputs[1];
    auto rstd = outputs[2];

    auto &output = outputs[0].value();

    auto num_groups = operation_attributes.num_groups;
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

    const bool is_lastdim_layernorm = false;
    const bool is_group_norm = true;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const auto mask_h = do_mask_h ? origin_h % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_w % TILE_WIDTH : TILE_WIDTH;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n * num_groups;
    const auto num_inner_tiles = (num_channels / num_groups) * Ht * Wt;

    const auto f_c = static_cast<float>(num_channels / num_groups);
    const auto f_ht = static_cast<float>(origin_h) / static_cast<float>(TILE_HEIGHT);
    const auto f_wt = static_cast<float>(origin_w) / static_cast<float>(TILE_WIDTH);
    auto scaler = 1.0f / (static_cast<float>(TILE_WIDTH) * sqrt(f_c * f_ht * f_wt));

    const bool gamma_has_value = gamma.has_value();
    const bool beta_has_value = beta.has_value();
    const bool mean_has_value = mean.has_value();
    const bool rstd_has_value = rstd.has_value();

    constexpr uint32_t MAX_BLOCK_SIZE = 8;
    const uint32_t block_size = get_block_size(num_inner_tiles, MAX_BLOCK_SIZE);

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
    const uint32_t in1_t = 1;                                 // scaler
    const uint32_t in2_t = 1;                                 // epsilon
    const uint32_t in3_t = gamma_has_value ? block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 1 : 0;                 // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;                 // mask_w

    const uint32_t out0_t = block_size;              // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    const uint32_t im0_t = 1;                                                         // E[x]
    uint32_t im1_t = num_inner_tiles;                                                 // x - E[x]
    uint32_t im2_t = 1;                                                               // (x - E[x])^2
    const uint32_t im3_t = 1;                                                         // Sum[(x - E[x])^2]
    const uint32_t im4_t = 1;                                                         // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 1;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                         // Sum[x]

    const auto cb_data_format = datatype_to_dataformat_converter(input.get_dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    const auto cb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + out0_t + out1_t + out2_t + im0_t +
                           im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) *
                          single_tile_size;
    const auto available_L1 = device->l1_size_per_core() - device->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large moreh_group_norm algorithm is selected.");
        in0_t = block_size;
        im1_t = 2 * block_size;
        im2_t = 2 * block_size;
    } else {
        log_info(LogTest, "Small moreh_group_norm algorithm is selected.");
    }

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},        // input
            {CB::c_in1, in1_t},        // scaler
            {CB::c_in2, in2_t},        // eps
            {CB::c_in3, in3_t},        // gamma
            {CB::c_in4, in4_t},        // beta
            {CB::c_in5, in5_t},        // mask_h
            {CB::c_in6, in6_t},        // mask_w
            {CB::c_out0, out0_t},      // output
            {CB::c_out1, out1_t},      // mean
            {CB::c_out2, out2_t},      // rstd
            {CB::c_intermed0, im0_t},  // E[x]
            {CB::c_intermed1, im1_t},  // x - E[x]
            {CB::c_intermed2, im2_t},  // (x - E[x])^2
            {CB::c_intermed3, im3_t},  // Sum[(x - E[x])^2]
            {CB::c_intermed4, im4_t},  // E[(x - E[x])^2] = Var[x]
            {CB::c_intermed5, im5_t},  // 1.0/(sqrt(Var[x] + eps))
            {CB::c_intermed6, im6_t},  // y * gamm + beta
            {CB::c_intermed7, im7_t},  // Sum[x]
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file = use_large_algorithm ? "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/"
                                                          "kernels/dataflow/reader_moreh_group_norm_large.cpp"
                                                        : "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/"
                                                          "kernels/dataflow/reader_moreh_group_norm_small.cpp";

    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/dataflow/writer_moreh_group_norm.cpp");

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const auto compute_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp";

    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        block_size,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_group_norm)};

    CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, compute_args_group_1}, compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,
            origin_h,
            origin_w,
            num_inner_tiles,
            block_size,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(beta_has_value),
            static_cast<uint32_t>(mean_has_value),
            static_cast<uint32_t>(rstd_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_group_norm)};

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

    const auto output_addr = output.buffer()->address();
    const auto mean_addr = mean_has_value ? mean.value().buffer()->address() : 0;
    const auto rstd_addr = rstd_has_value ? rstd.value().buffer()->address() : 0;

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;
    const auto beta_addr = beta_has_value ? beta.value().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            gamma_addr,
            static_cast<uint32_t>(is_dram(gamma)),
            static_cast<uint32_t>(gamma_has_value),
            beta_addr,
            static_cast<uint32_t>(is_dram(beta)),
            static_cast<uint32_t>(beta_has_value),
            *reinterpret_cast<uint32_t *>(&scaler),
            *reinterpret_cast<uint32_t *>(&eps),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
            num_channels,
            origin_h,
            origin_w,
            block_size,
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
            num_groups,
            block_size,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_rows_per_core * num_inner_tiles;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void MorehGroupNormOperation::MorehGroupNormFactory::override_runtime_arguments(
    cached_program_t &cached_program,
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &tensor_return_value) {
    auto input_buffer = tensor_args.input.buffer();
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
            auto &runtime_args = GetRuntimeArgs(cached_program.program, reader_kernels_id, core);
            runtime_args[0] = input_buffer->address();
            if (gamma_buffer != nullptr) {
                runtime_args[2] = gamma_buffer->address();
            }
            if (beta_buffer != nullptr) {
                runtime_args[5] = beta_buffer->address();
            }
        }

        {
            auto &runtime_args = GetRuntimeArgs(cached_program.program, writer_kernels_id, core);
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
}  // namespace ttnn::operations::moreh::moreh_group_norm
