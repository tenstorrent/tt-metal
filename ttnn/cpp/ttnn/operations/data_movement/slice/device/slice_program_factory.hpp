// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks slice_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step);

tt::tt_metal::operation::ProgramWithCallbacks slice_rm_multi_core_stride(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& slice_step);

tt::tt_metal::operation::ProgramWithCallbacks slice_multi_core_with_tensor_args(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

tt::tt_metal::operation::ProgramWithCallbacks slice_tile_multi_core_tensor_args(
    const Tensor& input_tensor, const Tensor& start_tensor, const Tensor& end_tensor, Tensor& output_tensor);

void set_slice_runtime_args_tensor_args(
    const Tensor& input_tensor,
    const Tensor& start_tensor,
    const Tensor& end_tensor,
    const Tensor& output_tensor,
    const uint32_t& num_cores_total,
    const uint32_t& num_cores,
    const std::vector<CoreCoord>& cores,
    const uint32_t& num_cores_group_1,
    const uint32_t& num_cores_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const Program& program,
    const tt::tt_metal::KernelHandle& unary_reader_kernel_id,
    const tt::tt_metal::KernelHandle& unary_writer_kernel_id);

}  // namespace ttnn::operations::data_movement::detail
