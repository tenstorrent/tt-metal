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

}  // namespace ttnn::operations::data_movement::detail
