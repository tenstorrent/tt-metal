// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks pad_rm_reader_writer(
    const Tensor& a,
    Tensor& output,
    const ttnn::SimpleShape& output_logical_shape,
    const ttnn::SimpleShape& input_tensor_start,
    const float pad_value);

tt::tt_metal::operation::ProgramWithCallbacks pad_tile(
    const Tensor& a,
    Tensor& output,
    const ttnn::SimpleShape& output_padded_shape,
    const ttnn::SimpleShape& input_tensor_start,
    const float pad_value);

tt::tt_metal::operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::SimpleShape& output_padded_shape,
    const ttnn::SimpleShape& input_tensor_start,
    const float pad_value);

tt::tt_metal::operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core_v2(
    const Tensor& a,
    Tensor& output,
    const ttnn::SimpleShape& output_padded_shape,
    const ttnn::SimpleShape& input_tensor_start,
    const float pad_value);

tt::tt_metal::operation::ProgramWithCallbacks pad_rm_sharded_height_only(
    const Tensor& a,
    Tensor& output,
    const ttnn::SimpleShape& output_padded_shape,
    const ttnn::SimpleShape& input_tensor_start,
    const float pad_value);

tt::tt_metal::operation::ProgramWithCallbacks pad_rm_sharded_width_only(
    const Tensor& a,
    Tensor& output,
    const ttnn::SimpleShape& output_padded_shape,
    const ttnn::SimpleShape& input_tensor_start,
    float pad_value);

}  // namespace ttnn::operations::data_movement::detail
