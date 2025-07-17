// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_layer_norm_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm {

inline uint32_t find_divisor_with_max_block_size(uint32_t val, uint32_t max_block_size) {
    uint32_t divisor{1};
    for (uint32_t current_divisor = max_block_size; current_divisor >= 1; current_divisor--) {
        if (val % current_divisor == 0) {
            divisor = current_divisor;
            break;
        }
    }
    return divisor;
}

MorehLayerNormOperation::ProgramFactory::cached_program_t MorehLayerNormOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& input = tensor_args.input;
    auto& gamma = tensor_args.gamma;
    auto& beta = tensor_args.beta;

    auto& mean_inp = tensor_args.mean;
    auto& rstd_inp = tensor_args.rstd;

    auto& output_tensors = output_tensor;
    const std::optional<const Tensor>& output = output_tensor.at(0);

    std::optional<Tensor> mean = std::nullopt;
    if (mean_inp.has_value()) {
        mean = output_tensor.at(1);
    }
    const std::optional<const Tensor> mean_as_tensor = mean ? std::optional<const Tensor>(mean.value()) : std::nullopt;

    std::optional<Tensor> rstd = std::nullopt;
    if (rstd_inp.has_value()) {
        rstd = output_tensor.at(2);
    }
    const std::optional<const Tensor> rstd_as_tensor = rstd ? std::optional<const Tensor>(rstd.value()) : std::nullopt;

    auto normalized_dims = operation_attributes.normalized_dims;
    auto eps = operation_attributes.eps;

    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    using namespace tt::constants;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = input.device();
    Program program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();
    const auto input_shape_without_padding = input.logical_shape();
    const auto input_rank = input_shape.rank();

    const bool is_lastdim_layer_norm = normalized_dims == 1;
    const bool is_groupnorm = false;

    auto num_inner = compute_inner(input_shape, normalized_dims);
    auto num_outer = compute_outer(input_shape, normalized_dims);

    const auto gamma_has_value = gamma.has_value();
    const auto beta_has_value = beta.has_value();
    const auto mean_has_value = mean.has_value();
    const auto rstd_has_value = rstd.has_value();

    const auto origin_H = input_shape_without_padding[-2];
    const auto origin_W = input_shape_without_padding[-1];

    uint32_t mean_rstd_height = 0;
    uint32_t mean_rstd_width = 0;

    if (mean_has_value) {
        const auto mean_rstd_shape_without_padding = mean->logical_shape();
        mean_rstd_height = mean_rstd_shape_without_padding[-2];
        mean_rstd_width = mean_rstd_shape_without_padding[-1];
    }

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layer_norm;
    const auto mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_W % TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_W % TILE_WIDTH : TILE_WIDTH;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    // core_group_2 works more.
    // If number of working cores is 108 and num_outer is 110,
    // core_group_2[(x=0, y=0), (x=0, y=1)] works for 2 rows. Others work for 1 row.
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_outer);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    // This could be inefficient.
    // If Wt is 65, the block_size will be 5. Then, the number of iteration is 13.
    // It can be 8 * 8 + 1, so the number of iterations is 9. It's more efficient.
    uint32_t MAX_BLOCK_SIZE = 4;
    if (fp32_dest_acc_en) {
        MAX_BLOCK_SIZE = 2;
    }
    const uint32_t block_size = find_divisor_with_max_block_size(num_inner, MAX_BLOCK_SIZE);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_t = num_inner;                                   // input
    const uint32_t in1_t = 1;                                     // scaler
    const uint32_t in2_t = 1;                                     // epsilon
    const uint32_t in3_t = gamma_has_value ? 2 * block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? 2 * block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 1 : 0;                     // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;                     // mask_w

    const uint32_t out0_t = 2 * block_size;          // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    const uint32_t im0_t = 1;                                                         // E[x]
    uint32_t im1_t = num_inner;                                                       // x - E[x]
    uint32_t im2_t = 1;                                                               // (x - E[x])^2
    const uint32_t im3_t = 1;                                                         // Sum[(x - E[x])^2]
    const uint32_t im4_t = 1;                                                         // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 1;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                         // Sum[x]

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const auto intermed_single_tile_size = tt::tt_metal::detail::TileSize(intermed_cb_format);

    const uint32_t cb_usage =
        (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + out0_t + out1_t + out2_t) * single_tile_size +
        (im0_t + im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) * intermed_single_tile_size;
    const uint32_t available_L1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(tt::LogTest, "Large moreh_layer_norm algorithm is selected.");
        in0_t = 2 * block_size;
        im1_t = 2 * block_size;
        im2_t = 2 * block_size;
    } else {
        log_info(tt::LogTest, "Small moreh_layer_norm algorithm is selected.");
    }

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {tt::CBIndex::c_0, in0_t},                       // input
            {tt::CBIndex::c_1, in1_t},                       // scaler
            {tt::CBIndex::c_2, in2_t},                       // epsilon
            {tt::CBIndex::c_3, in3_t},                       // gamma
            {tt::CBIndex::c_4, in4_t},                       // beta
            {tt::CBIndex::c_5, in5_t},                       // mask_h
            {tt::CBIndex::c_6, in6_t},                       // mask_w
            {tt::CBIndex::c_16, out0_t},                     // output
            {tt::CBIndex::c_17, out1_t},                     // mean
            {tt::CBIndex::c_18, out2_t},                     // rstd
            {tt::CBIndex::c_24, im0_t, intermed_cb_format},  // E[x]
            {tt::CBIndex::c_25, im1_t, intermed_cb_format},  // x - E[x]
            {tt::CBIndex::c_26, im2_t, intermed_cb_format},  // (x - E[x])^2
            {tt::CBIndex::c_27, im3_t, intermed_cb_format},  // Sum[(x - E[x])^2]
            {tt::CBIndex::c_28, im4_t, intermed_cb_format},  // E[(x - E[x])^2] = Var[x]
            {tt::CBIndex::c_29, im5_t, intermed_cb_format},  // 1.0/(sqrt(Var[x] + eps))
            {tt::CBIndex::c_30, im6_t, intermed_cb_format},  // y * gamm + beta
            {tt::CBIndex::c_31, im7_t, intermed_cb_format},  // Sum[x]
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(is_dram(gamma)),
        static_cast<uint32_t>(is_dram(beta)),
        block_size};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(is_dram(output)),
        static_cast<uint32_t>(is_dram(mean_as_tensor)),
        static_cast<uint32_t>(is_dram(rstd_as_tensor)),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value),
        block_size};

    std::map<std::string, std::string> reader_defines{};
    std::map<std::string, std::string> compute_defines{};
    if (gamma_has_value) {
        reader_defines["GAMMA_HAS_VALUE"] = "1";
    }
    if (beta_has_value) {
        reader_defines["BETA_HAS_VALUE"] = "1";
    }
    if (do_mask_h) {
        reader_defines["DO_MASK_H"] = "1";
    }
    if (do_mask_w) {
        reader_defines["DO_MASK_W"] = "1";
    }
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    if (is_lastdim_layer_norm) {
        compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";
    } else {
        compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";
    }
    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = "1";
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const auto reader_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/reader_moreh_layer_norm_large.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/reader_moreh_layer_norm_small.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/writer_moreh_layer_norm.cpp";

    const auto reader_kernels_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,
        origin_H,
        origin_W,
        num_inner,
        block_size,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value),
        static_cast<uint32_t>(is_lastdim_layer_norm),
        static_cast<uint32_t>(is_groupnorm)};

    const auto compute_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp";

    CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_rows_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,
            origin_H,
            origin_W,
            num_inner,
            block_size,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(beta_has_value),
            static_cast<uint32_t>(mean_has_value),
            static_cast<uint32_t>(rstd_has_value),
            static_cast<uint32_t>(is_lastdim_layer_norm),
            static_cast<uint32_t>(is_groupnorm)};

        CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_rows_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    union {
        float f;
        uint32_t u;
    } scaler;

    if (normalized_dims == 1) {
        scaler.f = 1.0f / static_cast<float>(origin_W);
    } else {
        auto reduce_size = 1;
        for (uint32_t i = input_rank - normalized_dims; i < input_rank; i++) {
            auto size = input_shape_without_padding[i];
            reduce_size *= size;
        }

        scaler.f = 1.0f / static_cast<float>(sqrt(reduce_size));
    }

    union {
        float f;
        uint32_t u;
    } e;
    e.f = eps;  // epsilon

    const auto input_addr = input.buffer()->address();
    const auto output_addr = output->buffer()->address();

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;
    const auto beta_addr = beta_has_value ? beta.value().buffer()->address() : 0;
    const auto mean_addr = mean_has_value ? mean.value().buffer()->address() : 0;
    const auto rstd_addr = rstd_has_value ? rstd.value().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            gamma_addr,
            beta_addr,
            num_rows_per_core,
            num_inner,
            tile_offset,
            scaler.u,
            e.u,
            mask_h,
            mask_w};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{
            output_addr,
            mean_addr,
            rstd_addr,
            num_rows_per_core,
            num_inner,
            tile_offset,
            mean_rstd_height,
            mean_rstd_width,
            normalized_dims};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_rows_per_core * num_inner;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores, num_cores_y}};
}

void MorehLayerNormOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    auto input_buffer = tensor_args.input.buffer();
    auto gamma_buffer = tensor_args.gamma.has_value() ? tensor_args.gamma->buffer() : nullptr;
    auto beta_buffer = tensor_args.beta.has_value() ? tensor_args.beta->buffer() : nullptr;
    auto mean_buffer = tensor_args.mean.has_value() ? tensor_args.mean->buffer() : nullptr;
    auto rstd_buffer = tensor_args.rstd.has_value() ? tensor_args.rstd->buffer() : nullptr;

    auto output_buffer = tensor_return_value.at(0)->buffer();

    auto num_cores = cached_program.shared_variables.num_cores;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = input_buffer->address();
            if (gamma_buffer != nullptr) {
                runtime_args[1] = gamma_buffer->address();
            }
            if (beta_buffer != nullptr) {
                runtime_args[2] = beta_buffer->address();
            }
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_buffer->address();
            if (mean_buffer != nullptr) {
                runtime_args[1] = mean_buffer->address();
            }
            if (rstd_buffer != nullptr) {
                runtime_args[2] = rstd_buffer->address();
            }
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm
