// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "moreh_mean_backward_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

using namespace tt::tt_metal;

void get_tensor_dim(ttnn::SmallVector<uint32_t>& dim, const ttnn::Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = (shape[idx] + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
        } else {
            dim[i] = shape[idx];
        }
    }
}

ttnn::Shape get_output_grad_shape(
    const Tensor& output_grad, const Tensor& input_grad, const ttnn::SmallVector<int64_t>& dims, const bool& keepdim) {
    if (keepdim) {
        return output_grad.logical_shape();
    }

    auto shape = input_grad.logical_shape();
    auto rank = shape.rank();
    for (auto dim : dims) {
        TT_FATAL(dim < rank, "dim {} < rank {}", dim, rank);
        shape[dim] = 1;
    }

    return shape;
}

namespace ttnn::operations::moreh::moreh_mean_backward {

MorehMeanBackwardOperation::MorehMeanBackwardFactory::cached_program_t
MorehMeanBackwardOperation::MorehMeanBackwardFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input_grad = output;
    auto keepdim = operation_attributes.keepdim;
    auto dims = operation_attributes.dims;
    auto compute_kernel_config =
        init_device_compute_kernel_config(output_grad.device()->arch(), operation_attributes.compute_kernel_config);
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = output_grad.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());

    const auto& input_grad_shape = input_grad.logical_shape();
    const uint32_t input_grad_rank = input_grad_shape.rank();

    ttnn::SmallVector<uint32_t> input_grad_dim(input_grad_rank, 1);
    get_tensor_dim(input_grad_dim, input_grad_shape);
    const auto& output_grad_shape = get_output_grad_shape(output_grad, input_grad, dims, keepdim);

    ttnn::SmallVector<uint32_t> output_grad_dim(input_grad_rank, 1);
    get_tensor_dim(output_grad_dim, output_grad_shape);

    ttnn::SmallVector<uint32_t> need_bcast_dim(input_grad_rank, 0);
    for (auto i = 0; i < input_grad_rank; ++i) {
        auto idx = input_grad_rank - 1 - i;
        bool is_tile_dim = (idx == input_grad_rank - 1 || idx == input_grad_rank - 2);

        if (is_tile_dim) {
            need_bcast_dim[i] = (output_grad_shape[idx] != input_grad_shape[idx]);
        } else {
            need_bcast_dim[i] = (output_grad_shape[idx] != input_grad_shape[idx]);
        }
    }
    const auto num_input_grad_tiles = input_grad.physical_volume() / tt::constants::TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(output_grad.device()->arch(), compute_kernel_config);

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
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {tt::CBIndex::c_0, 2},  // input
            {tt::CBIndex::c_1, 1},  // zero
            {tt::CBIndex::c_2, 1},  // scalar
            {tt::CBIndex::c_24, 1},
            {tt::CBIndex::c_16, 2},  // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(is_dram(output_grad)), input_grad_rank};
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(is_dram(input_grad))};
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_mean_backward/device/kernels/reader_moreh_mean_backward.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_mean_backward/device/kernels/writer_moreh_mean_backward.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_mean_backward/device/kernels/moreh_mean_backward.cpp";
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1, need_bcast_dim[0], need_bcast_dim[1]};
    const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2, need_bcast_dim[0], need_bcast_dim[1]};
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    auto compute_kernel_ids = CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, num_cols_per_core_group_1, compute_args_group_1},
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
        },
        ComputeKernelConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .defines = compute_defines});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    uint32_t num_dim = 1;
    for (auto dim : dims) {
        num_dim *= input_grad_shape[dim];
    }

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        std::vector<uint32_t> reader_rt_args;
        reader_rt_args.push_back(output_grad.buffer()->address());
        reader_rt_args.push_back(num_tiles_per_core);
        reader_rt_args.push_back(tile_offset);
        reader_rt_args.push_back(num_dim);
        reader_rt_args.insert(reader_rt_args.end(), output_grad_dim.begin(), output_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), input_grad_dim.begin(), input_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), need_bcast_dim.begin(), need_bcast_dim.end());

        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        SetRuntimeArgs(
            program, writer_kernel_id, core, {input_grad.buffer()->address(), num_tiles_per_core, tile_offset});

        tile_offset += num_tiles_per_core;
    }
    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y}};
}

void MorehMeanBackwardOperation::MorehMeanBackwardFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    auto output_grad_address = tensor_args.output_grad.buffer()->address();
    auto input_grad_address = output.buffer()->address();
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = output_grad_address;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = input_grad_address;
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_mean_backward
