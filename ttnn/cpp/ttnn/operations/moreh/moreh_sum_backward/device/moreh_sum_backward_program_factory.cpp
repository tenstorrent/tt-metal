// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "moreh_sum_backward_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_sum_backward {

using namespace tt::tt_metal;

void get_tensor_dim(ttnn::SmallVector<uint32_t>& dim, const ttnn::Shape& padded_shape) {
    const auto rank = padded_shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = padded_shape[idx] / tt::constants::TILE_HEIGHT;
        } else {
            dim[i] = padded_shape[idx];
        }
    }

    log_debug(tt::LogOp, "rank {}", rank);
    for (auto i = 0; i < rank; ++i) {
        log_debug(tt::LogOp, "dim[{}] = {}", i, dim[i]);
    }
}

std::pair<ttnn::Shape, ttnn::Shape> get_output_grad_shape(
    const Tensor& output_grad, const Tensor& input_grad, const ttnn::SmallVector<int64_t>& dims, const bool& keepdim) {
    if (keepdim) {
        return {output_grad.logical_shape(), output_grad.padded_shape()};
    }

    auto logical_shape = input_grad.logical_shape();
    auto padded_shape = input_grad.padded_shape();
    auto rank = logical_shape.rank();
    for (auto dim : dims) {
        TT_FATAL(dim < rank, "dim {} < rank {}", dim, rank);
        bool is_tile_dim = (dim == rank - 1 || dim == rank - 2);
        logical_shape[dim] = 1;
        if (is_tile_dim) {
            padded_shape[dim] = tt::constants::TILE_HEIGHT;
        } else {
            padded_shape[dim] = 1;
        }
    }

    return {logical_shape, padded_shape};
}

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward/device/kernels/reader_moreh_sum_backward.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward/device/kernels/writer_moreh_sum_backward.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward/device/kernels/moreh_sum_backward.cpp";

ProgramDescriptor MorehSumBackwardOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input_grad = output_tensor;

    const auto& dims = operation_attributes.dims;
    auto keepdim = operation_attributes.keepdim;
    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    const auto& input_grad_shape = input_grad.padded_shape();
    const auto& input_grad_shape_wo_padding = input_grad.logical_shape();
    const uint32_t input_grad_rank = input_grad_shape.rank();

    ttnn::SmallVector<uint32_t> input_grad_dim(input_grad_rank, 1);
    log_debug(tt::LogOp, "input_grad");
    get_tensor_dim(input_grad_dim, input_grad_shape);
    const auto [output_grad_shape_wo_padding, output_grad_shape] =
        get_output_grad_shape(output_grad, input_grad, dims, keepdim);

    ttnn::SmallVector<uint32_t> output_grad_dim(input_grad_rank, 1);
    log_debug(tt::LogOp, "output_grad");
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
    const auto num_input_grad_tiles = input_grad.physical_volume() / tt::constants::TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(output_grad.device()->arch(), compute_kernel_config);

    for (auto i = 0; i < input_grad_rank; ++i) {
        log_debug(tt::LogOp, "need_bcast_dim [{}] = {}", i, need_bcast_dim[i]);
    }
    log_debug(tt::LogOp, "num_input_grad_tiles {}", num_input_grad_tiles);
    log_debug(
        tt::LogOp,
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
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            split_work_to_cores(grid, num_input_grad_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });  // input
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });  // zero
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });  // output

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_ct_args = {input_grad_rank};
    TensorAccessorArgs(output_grad.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores);

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(input_grad.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    ComputeConfigDescriptor compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // Compute kernel for core_group_1
    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {num_cols_per_core_group_1, need_bcast_dim[0], need_bcast_dim[1]};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = compute_config;

    // Compute kernel for core_group_2 (may be empty)
    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = COMPUTE_KERNEL_PATH;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {num_cols_per_core_group_2, need_bcast_dim[0], need_bcast_dim[1]};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = compute_config;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // Build reader runtime args: addr, num_tiles, offset, then dim vectors
        KernelDescriptor::CoreRuntimeArgs reader_rt_args;
        reader_rt_args.push_back(output_grad.buffer()->address());
        reader_rt_args.push_back(num_tiles_per_core);
        reader_rt_args.push_back(tile_offset);
        reader_rt_args.insert(reader_rt_args.end(), output_grad_dim.begin(), output_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), input_grad_dim.begin(), input_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), need_bcast_dim.begin(), need_bcast_dim.end());

        reader_desc.runtime_args.emplace_back(core, std::move(reader_rt_args));

        writer_desc.emplace_runtime_args(core, {input_grad.buffer(), num_tiles_per_core, tile_offset});

        tile_offset += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_sum_backward
