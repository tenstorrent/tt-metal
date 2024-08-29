// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
namespace tt {
using namespace constants;
namespace operations {

namespace primary {

namespace {

void get_tensor_dim(std::vector<uint32_t> &dim, const Shape &shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = shape[idx] / TILE_HEIGHT;
        } else {
            dim[i] = shape[idx];
        }
    }

    log_debug(LogOp, "rank {}", rank);
    for (auto i = 0; i < rank; ++i) {
        log_debug(LogOp, "dim[{}] = {}", i, dim[i]);
    }
}

Shape get_output_grad_shape(
    const Tensor &output_grad, const Tensor &input_grad, const std::vector<int64_t> &dims, const bool &keep_batch_dim) {
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
            shape[dim] = TILE_HEIGHT;
            padding[dim] = Padding::PadDimension{0, 31};
        } else {
            shape[dim] = 1;
        }
    }

    return Shape(shape, padding);
}
}  // namespace

operation::ProgramWithCallbacks moreh_sum_backward_impl(
    const Tensor &output_grad,
    const Tensor &input_grad,
    const std::vector<int64_t> &dims,
    const bool &keep_batch_dim,
    const DeviceComputeKernelConfig &compute_kernel_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto *device = output_grad.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.get_dtype());
    const auto single_tile_size = detail::TileSize(cb_data_format);

    const auto &input_grad_shape = input_grad.get_legacy_shape();
    const auto &input_grad_shape_wo_padding = input_grad_shape.without_padding();
    const uint32_t input_grad_rank = input_grad_shape.rank();

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
    const auto num_input_grad_tiles = input_grad.volume() / TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(output_grad.device()->arch(), compute_kernel_config);

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
         num_cols_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_input_grad_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, 2},   // input
            {CB::c_in1, 1},   // zero
            {CB::c_out0, 2},  // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(is_dram(output_grad)), input_grad_rank};
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(is_dram(input_grad))};
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_impl/kernels/"
        "reader_moreh_sum_backward.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_impl/kernels/"
        "writer_moreh_sum_backward.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1, need_bcast_dim[0], need_bcast_dim[1]};
    std::map<string, string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_impl/kernels/"
        "moreh_sum_backward.cpp";
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
        const std::vector<uint32_t> compute_args_group_2{
            num_cols_per_core_group_2, need_bcast_dim[0], need_bcast_dim[1]};
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

        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        std::vector<uint32_t> reader_rt_args;
        reader_rt_args.push_back(output_grad.buffer()->address());
        reader_rt_args.push_back(num_tiles_per_core);
        reader_rt_args.push_back(tile_offset);
        reader_rt_args.insert(reader_rt_args.end(), output_grad_dim.begin(), output_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), input_grad_dim.begin(), input_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), need_bcast_dim.begin(), need_bcast_dim.end());

        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        SetRuntimeArgs(
            program, writer_kernel_id, core, {input_grad.buffer()->address(), num_tiles_per_core, tile_offset});

        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        const auto *output_grad_buffer = input_tensors.at(0).buffer();
        const auto *input_grad_buffer = output_tensors.at(0).buffer();
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = output_grad_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = input_grad_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
