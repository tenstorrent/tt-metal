// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ttnn::operations::data_movement::detail {

const std::map<ttnn::DataType, uint32_t> data_type_to_size = {
    {ttnn::DataType::BFLOAT16, 2},
    {ttnn::DataType::FLOAT32, 4},
    {ttnn::DataType::UINT16, 2},
    {ttnn::DataType::UINT32, 4},
    {ttnn::DataType::INT32, 4},
    {ttnn::DataType::UINT8, 1},
};

tt::tt_metal::operation::ProgramWithCallbacks fill_pad_multi_core(const Tensor& input_tensor, float fill_value);

}  // namespace ttnn::operations::data_movement::detail
