// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_norm_backward/moreh_norm_backward_op.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"
#include "ttnn/deprecated/tt_numpy/functions.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {
std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    const bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}


void get_tensor_dim(std::vector<uint32_t> &dim, const Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = shape[idx] / tt::constants::TILE_HEIGHT;
        }
        else {
            dim[i] = shape[idx];
        }
    }

    log_debug(LogOp, "rank {}", rank);
    for (auto i = 0; i < rank; ++i) {
        log_debug(LogOp, "dim[{}] = {}", i, dim[i]);
    }
}

Shape get_output_grad_shape(const Tensor &output_grad, const Tensor &input_grad, const std::vector<int64_t> &dims, const bool &keep_batch_dim) {
    if (keep_batch_dim) {
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

    return Shape(shape, padding);
}

}  // namespace

operation::ProgramWithCallbacks moreh_norm_backward_(
    const Tensor &input, const Tensor &output, const Tensor &output_grad, float p, const std::vector<int64_t> &dims, const bool &keep_batch_dim, const Tensor &input_grad, const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = output_grad.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto &input_grad_shape = input_grad.get_legacy_shape();
    const auto &input_grad_shape_wo_padding = input_grad_shape.without_padding();
    const auto input_grad_rank = input_grad_shape.rank();

    std::vector<uint32_t> input_grad_dim(input_grad_rank, 1);
    log_debug(LogOp, "input_grad");
    get_tensor_dim(input_grad_dim, input_grad_shape);
    const auto &output_grad_shape = get_output_grad_shape(output_grad, input_grad, dims, keep_batch_dim);
    const auto &output_grad_shape_wo_padding = output_grad_shape.without_padding();

    std::vector<uint32_t> output_grad_dim(input_grad_rank, 1);
    log_debug(LogOp, "output_grad");
    get_tensor_dim(output_grad_dim, output_grad_shape);

    std::vector<uint32_t> need_bcast_dim(input_grad_rank, 0);
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
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(output_grad.device()->arch(), compute_kernel_config);

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
    auto [floored_p_minus_one, decimal_minus_one, p_minus_one_is_negative] =
        get_floored_p_and_decimal_and_p_is_negative(p - 1.0f);

    TT_ASSERT(tt::numpy::detail::nearly_equal(decimal_minus_one, decimal));

    for (auto i = 0; i < input_grad_rank; ++i) {
        log_debug(LogOp, "need_bcast_dim [{}] = {}", i, need_bcast_dim[i]);
    }
    log_debug(LogOp, "num_input_grad_tiles {}", num_input_grad_tiles);
    log_debug(
        LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

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
         num_cols_per_core_group_2] = ttnn::operations::core::work_split::split_work_to_cores(grid, num_input_grad_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(output_grad.get_dtype());
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
            {CB::c_in0, in0_t},    // input
            {CB::c_in1, in1_t},    // output
            {CB::c_in2, in2_t},    // output_grad
            {CB::c_in3, in3_t},    // decimal
            {CB::c_out0, out0_t},  // input_grad
            {CB::c_intermed0, im0_t, intermed_data_format},
            {CB::c_intermed1, im1_t, intermed_data_format},
            {CB::c_intermed2, im2_t, intermed_data_format},
            {CB::c_intermed3, im3_t, intermed_data_format},
            {CB::c_intermed4, im4_t, intermed_data_format},
            {CB::c_intermed5, im5_t, intermed_data_format},
            {CB::c_intermed6, im6_t, intermed_data_format},
            {CB::c_intermed7, im7_t, intermed_data_format},
        });


    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_norm_backward/kernels/"
        "reader_moreh_norm_backward.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_norm_backward/kernels/"
        "writer_moreh_norm_backward.cpp";

    std::vector<uint32_t> reader_compile_time_args =
    { static_cast<uint32_t>(is_dram(input)), static_cast<uint32_t>(is_dram(output)), static_cast<uint32_t>(is_dram(output_grad)), static_cast<uint32_t>(input_grad_rank) };
    std::vector<uint32_t> writer_compile_time_args =
    { static_cast<uint32_t>(is_dram(input_grad)) };
    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_norm_backward/kernels/"
        "moreh_norm_backward_kernel.cpp";
    std::map<std::string, std::string> compute_defines{};
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1, need_bcast_dim[0], need_bcast_dim[1]};
    const auto compute_kernels_id_1 =
        CreateComputeKernel(program, compute_kernel_file, {core_group_1, num_cols_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    KernelHandle compute_kernels_id_2{0};
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2, need_bcast_dim[0], need_bcast_dim[1]};
        compute_kernels_id_2 =
            CreateComputeKernel(program, compute_kernel_file, {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();
    const auto output_addr = output.buffer()->address();
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_grad_addr = input_grad.buffer()->address();

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        KernelHandle compute_kernel_id;
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
            compute_kernel_id = compute_kernels_id_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
            compute_kernel_id = compute_kernels_id_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        std::vector<uint32_t> reader_rt_args{
            input_addr,
            output_addr,
            output_grad_addr,
            *reinterpret_cast<uint32_t *>(&decimal),
            num_tiles_per_core,
            tile_offset};
        reader_rt_args.insert(reader_rt_args.end(), output_grad_dim.begin(), output_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), input_grad_dim.begin(), input_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), need_bcast_dim.begin(), need_bcast_dim.end());

        SetRuntimeArgs(program, reader_kernels_id, core, reader_rt_args);

        // writer
        std::vector<uint32_t> writer_runtime_args{
            input_grad_addr,
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

    return {
        .program = std::move(program),
        .override_runtime_arguments_callback =
            create_override_runtime_arguments_callback(reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y)};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
