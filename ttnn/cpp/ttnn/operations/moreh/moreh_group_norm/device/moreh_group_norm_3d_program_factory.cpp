// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_group_norm {
MorehGroupNormOperation::MorehGroupNorm3DFactory::cached_program_t
MorehGroupNormOperation::MorehGroupNorm3DFactory::create(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &outputs) {
    using namespace tt;
    using namespace tt::constants;
    using namespace tt::operations::primary;

    const auto &input = tensor_args.input;
    auto gamma = tensor_args.gamma;
    auto beta = tensor_args.beta;
    const std::optional<const Tensor> mean = outputs[1];
    const std::optional<const Tensor> rstd = outputs[2];

    auto &output = outputs[0].value();

    auto num_groups = operation_attributes.num_groups;
    auto eps = operation_attributes.eps;
    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.get_logical_shape();

    const auto n = input_shape[0];
    const auto c = input_shape[1];
    const auto w = input_shape[2];

    const bool gamma_has_value = gamma.has_value();
    const bool beta_has_value = beta.has_value();
    const bool mean_has_value = mean.has_value();
    const bool rstd_has_value = rstd.has_value();

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();

    const auto num_cores_y = grid.y;

    const auto num_units = n * num_groups;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_units_per_core_group_1,
         num_units_per_core_group_2] = split_work_to_cores(grid, num_units);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_units_per_core_group_1: {}", num_units_per_core_group_1);
    log_debug(LogTest, "num_units_per_core_group_2: {}", num_units_per_core_group_2);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_t = 1;                              // input
    const uint32_t in1_t = 1;                        // scaler
    const uint32_t in2_t = 1;                        // epsilon
    const uint32_t in3_t = gamma_has_value ? 1 : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? 1 : 0;   // beta
    const uint32_t in5_t = 1;                        // mask_h
    const uint32_t in6_t = 1;                        // mask_w
    const uint32_t in7_t = 1;                        // zeros

    const uint32_t out0_t = 1;                       // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    const uint32_t im0_t = 1;                                                // E[x]
    uint32_t im1_t = 1;                                                      // x - E[x]
    uint32_t im2_t = 1;                                                      // (x - E[x])^2
    const uint32_t im3_t = 1;                                                // Sum[(x - E[x])^2]
    const uint32_t im4_t = 1;                                                // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 1;                                                // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * 1 : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                // Sum[x]

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    const auto intermed_single_tile_size = tt_metal::detail::TileSize(intermed_data_format);

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},                              // input
            {CB::c_in1, in1_t, intermed_data_format},        // scaler
            {CB::c_in2, in2_t, intermed_data_format},        // eps
            {CB::c_in3, in3_t},                              // gamma
            {CB::c_in4, in4_t},                              // beta
            {CB::c_in5, in5_t},                              // mask_h
            {CB::c_in6, in6_t},                              // mask_w
            {CB::c_in7, in7_t, intermed_data_format},        // zeros
            {CB::c_out0, out0_t},                            // output
            {CB::c_out1, out1_t},                            // mean
            {CB::c_out2, out2_t},                            // rstd
            {CB::c_intermed0, im0_t, intermed_data_format},  // E[x]
            {CB::c_intermed1, im1_t, intermed_data_format},  // x - E[x]
            {CB::c_intermed2, im2_t, intermed_data_format},  // (x - E[x])^2
            {CB::c_intermed3, im3_t, intermed_data_format},  // Sum[(x - E[x])^2]
            {CB::c_intermed4, im4_t, intermed_data_format},  // E[(x - E[x])^2] = Var[x]
            {CB::c_intermed5, im5_t, intermed_data_format},  // 1.0/(sqrt(Var[x] + eps))
            {CB::c_intermed6, im6_t, intermed_data_format},  // y * gamm + beta
            {CB::c_intermed7, im7_t, intermed_data_format},  // Sum[x]
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(is_dram(gamma)),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(is_dram(beta))};
    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(is_dram(output)),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(is_dram(mean)),
        static_cast<uint32_t>(rstd_has_value),
        static_cast<uint32_t>(is_dram(rstd))};

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/reader_moreh_group_norm_3d.cpp";

    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/writer_moreh_group_norm_3d.cpp";

    std::map<string, string> reader_defines{};
    std::map<string, string> writer_defines{};

    const auto reader_kernels_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernels_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/moreh_group_norm_3d.cpp";

    const std::vector<uint32_t> compute_args_group_1{
        num_units_per_core_group_1,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value)};

    const std::vector<uint32_t> compute_args_group_2{
        num_units_per_core_group_2,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value)};

    vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    auto compute_kernel_ids = CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, num_units_per_core_group_1, compute_args_group_1},
            {core_group_2, num_units_per_core_group_2, compute_args_group_2},
        },
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .defines = compute_defines});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();

    const auto output_addr = output.buffer()->address();
    const auto mean_addr = mean_has_value ? mean.value().buffer()->address() : 0;
    const auto rstd_addr = rstd_has_value ? rstd.value().buffer()->address() : 0;

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;
    const auto beta_addr = beta_has_value ? beta.value().buffer()->address() : 0;

    union {
        float f;
        uint32_t u;
    } scaler;

    scaler.f = 1.0f / sqrt(static_cast<float>(c) / num_groups * w);

    for (uint32_t i = 0, unit_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_units_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_units_per_core = num_units_per_core_group_1;
            SetRuntimeArgs(program, compute_kernel_ids[0], core, {c, w, num_groups, unit_offset});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_units_per_core = num_units_per_core_group_2;
            SetRuntimeArgs(program, compute_kernel_ids[1], core, {c, w, num_groups, unit_offset});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            gamma_addr,
            beta_addr,
            scaler.u,
            *reinterpret_cast<uint32_t *>(&eps),
            unit_offset,
            num_units_per_core,
            c,
            w,
            num_groups,
        };
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            output_addr,
            mean_addr,
            rstd_addr,
            unit_offset,
            num_units_per_core,
            c,
            w,
            num_groups,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        unit_offset += num_units_per_core;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void MorehGroupNormOperation::MorehGroupNorm3DFactory::override_runtime_arguments(
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
                runtime_args[1] = gamma_buffer->address();
            }
            if (beta_buffer != nullptr) {
                runtime_args[2] = beta_buffer->address();
            }
        }

        {
            auto &runtime_args = GetRuntimeArgs(cached_program.program, writer_kernels_id, core);
            runtime_args[0] = ouput_buffer->address();
            if (mean_buffer != nullptr) {
                runtime_args[1] = mean_buffer->address();
            }
            if (rstd_buffer != nullptr) {
                runtime_args[2] = rstd_buffer->address();
            }
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_group_norm
