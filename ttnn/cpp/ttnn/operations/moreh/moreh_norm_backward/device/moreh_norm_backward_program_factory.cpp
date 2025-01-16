// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_norm_backward {

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}

void get_tensor_dim(ttnn::SmallVector<uint32_t>& dim, const tt::tt_metal::LegacyShape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = shape[idx] / tt::constants::TILE_HEIGHT;
        } else {
            dim[i] = shape[idx];
        }
    }
}

tt::tt_metal::LegacyShape get_output_grad_shape(
    const Tensor& output_grad, const Tensor& input_grad, const ttnn::SmallVector<int64_t>& dims, const bool& keepdim) {
    if (keepdim) {
        return output_grad.get_legacy_shape();
    }

    auto shape = input_grad.get_legacy_shape();
    auto rank = shape.rank();
    auto padding = shape.padding();
    for (auto dim : dims) {
        TT_FATAL(dim < rank, "dim {} < rank {}", dim, rank);
        bool is_tile_dim = (dim == rank - 1 || dim == rank - 2);
        if (is_tile_dim) {
            shape[dim] = tt::constants::TILE_HEIGHT;
            padding[dim] = Padding::PadDimension{0, 31};
        } else {
            shape[dim] = 1;
        }
    }
    return tt::tt_metal::LegacyShape(shape, padding);
}

MorehNormBackwardOperation::ProgramFactory::cached_program_t MorehNormBackwardOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input_grad) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;
    const auto& output_grad = tensor_args.output_grad;
    const auto p = operation_attributes.p;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = output_grad.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto& input_grad_shape = input_grad.get_legacy_shape();
    const auto& input_grad_shape_wo_padding = input_grad_shape.without_padding();
    const auto input_grad_rank = input_grad_shape.rank();

    ttnn::SmallVector<uint32_t> input_grad_dim(input_grad_rank, 1);
    get_tensor_dim(input_grad_dim, input_grad_shape);
    tt::tt_metal::LegacyShape output_grad_shape =
        get_output_grad_shape(output_grad, input_grad, operation_attributes.dims, operation_attributes.keepdim);
    const auto output_grad_shape_wo_padding = output_grad_shape.without_padding();

    ttnn::SmallVector<uint32_t> output_grad_dim(input_grad_rank, 1);
    get_tensor_dim(output_grad_dim, output_grad_shape);

    ttnn::SmallVector<uint32_t> need_bcast_dim(input_grad_rank, 0);
    for (auto i = 0; i < input_grad_rank; ++i) {
        auto idx = input_grad_rank - 1 - i;
        bool is_tile_dim = (idx == input_grad_rank - 1 || idx == input_grad_rank - 2);

        if (is_tile_dim) {
            need_bcast_dim[i] = (output_grad_shape_wo_padding[idx] != input_grad_shape_wo_padding[idx]);
        } else {
            need_bcast_dim[i] = (output_grad_shape[idx] != input_grad_shape[idx]);
        }
    }

    const auto num_input_grad_tiles = input_grad.volume() / tt::constants::TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(output_grad.device()->arch(), operation_attributes.compute_kernel_config);

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
    auto [floored_p_minus_one, decimal_minus_one, p_minus_one_is_negative] =
        get_floored_p_and_decimal_and_p_is_negative(p - 1.0f);

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
         num_cols_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_input_grad_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_grad.get_dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;

    const uint32_t in0_t{1};  // input(==x)
    const uint32_t in1_t{1};  // output(==y)
    const uint32_t in2_t{1};  // output_grad(==dy)
    const uint32_t in3_t{1};  // decimal

    // (x^(p - 1) * y * dy) / y^p
    const uint32_t out0_t{1};  // input_grad(==dx)

    const uint32_t im0_t{1};
    const uint32_t im1_t{1};
    const uint32_t im2_t{1};
    const uint32_t im3_t{1};
    const uint32_t im4_t{1};
    const uint32_t im5_t{1};
    const uint32_t im6_t{1};
    const uint32_t im7_t{1};

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {tt::CBIndex::c_0, in0_t},    // input
            {tt::CBIndex::c_1, in1_t},    // output
            {tt::CBIndex::c_2, in2_t},    // output_grad
            {tt::CBIndex::c_3, in3_t},    // decimal
            {tt::CBIndex::c_16, out0_t},  // input_grad
            {tt::CBIndex::c_24, im0_t, intermed_data_format},
            {tt::CBIndex::c_25, im1_t, intermed_data_format},
            {tt::CBIndex::c_26, im2_t, intermed_data_format},
            {tt::CBIndex::c_27, im3_t, intermed_data_format},
            {tt::CBIndex::c_28, im4_t, intermed_data_format},
            {tt::CBIndex::c_29, im5_t, intermed_data_format},
            {tt::CBIndex::c_30, im6_t, intermed_data_format},
            {tt::CBIndex::c_31, im7_t, intermed_data_format},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/"
        "reader_moreh_norm_backward.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/"
        "writer_moreh_norm_backward.cpp";

    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(is_dram(output)),
        static_cast<uint32_t>(is_dram(output_grad)),
        static_cast<uint32_t>(input_grad_rank)};
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(is_dram(input_grad))};
    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/"
        "moreh_norm_backward_kernel.cpp";
    std::map<std::string, std::string> compute_defines{};
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1, need_bcast_dim[0], need_bcast_dim[1]};
    const auto compute_kernels_id_1 = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_cols_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    KernelHandle compute_kernels_id_2{0};
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_cols_per_core_group_2, need_bcast_dim[0], need_bcast_dim[1]};
        compute_kernels_id_2 = CreateComputeKernel(
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

        KernelHandle compute_kernel_id;
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
            compute_kernel_id = compute_kernels_id_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
            compute_kernel_id = compute_kernels_id_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        std::vector<uint32_t> reader_rt_args{
            input.buffer()->address(),
            output.buffer()->address(),
            output_grad.buffer()->address(),
            *reinterpret_cast<uint32_t*>(&decimal),
            num_tiles_per_core,
            tile_offset};
        reader_rt_args.insert(reader_rt_args.end(), output_grad_dim.begin(), output_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), input_grad_dim.begin(), input_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), need_bcast_dim.begin(), need_bcast_dim.end());

        SetRuntimeArgs(program, reader_kernels_id, core, reader_rt_args);

        // writer
        std::vector<uint32_t> writer_runtime_args{
            input_grad.buffer()->address(),
            num_tiles_per_core,
            tile_offset,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{
            num_tiles_per_core,
            floored_p,
            static_cast<uint32_t>(p_is_negative),
            floored_p_minus_one,
            static_cast<uint32_t>(p_minus_one_is_negative)};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset += num_tiles_per_core;
    }

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void MorehNormBackwardOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto& writer_kernels_id = cached_program.shared_variables.writer_kernels_id;
    auto& num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    for (uint32_t icore = 0; icore < num_cores_to_be_used; icore++) {
        CoreCoord core = {icore / num_cores_y, icore % num_cores_y};
        // readers
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
            runtime_args[0] = tensor_args.input.buffer()->address();
            runtime_args[1] = tensor_args.output.buffer()->address();
            runtime_args[2] = tensor_args.output_grad.buffer()->address();
        }

        // writer
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
            runtime_args[0] = output.buffer()->address();
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_norm_backward
