// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"

namespace ttnn::operations::data_movement::detail {


operation::ProgramWithCallbacks pad_rm_reader_writer(const Tensor &a,
                                                     Tensor &output,
                                                     const tt::tt_metal::Shape &output_tensor_shape,
                                                     const tt::tt_metal::Shape &input_tensor_start,
                                                     const float pad_value);


operation::ProgramWithCallbacks pad_rm_opt(const Tensor &a,
                                           Tensor &output,
                                           const Shape &output_tensor_shape,
                                           const Shape &input_tensor_start,
                                           const float pad_value);

operation::ProgramWithCallbacks pad_rm(const Tensor &a, Tensor &output, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value);

operation::ProgramWithCallbacks pad_tile(const Tensor &a, Tensor& output, const tt::tt_metal::Shape &output_tensor_shape, const tt::tt_metal::Shape &input_tensor_start, const float pad_value);

operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core(const Tensor &a,
                                                                Tensor &output,
                                                                const tt::tt_metal::Shape &output_tensor_shape,
                                                                const tt::tt_metal::Shape &input_tensor_start,
                                                                const float pad_value);



operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core_v2(const Tensor &a,
                                                                Tensor &output,
                                                                const tt::tt_metal::Shape &output_tensor_shape,
                                                                const tt::tt_metal::Shape &input_tensor_start,
                                                                const float pad_value);


operation::ProgramWithCallbacks pad_rm_sharded(const Tensor &a,
                                                Tensor &output,
                                                const tt::tt_metal::Shape &output_tensor_shape,
                                                const tt::tt_metal::Shape &input_tensor_start,
                                                const float pad_value);


} // namespace ttnn::operations::reduction::detail
