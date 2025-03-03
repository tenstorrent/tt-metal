// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks fill_pad_multi_core(const Tensor& input_tensor, float fill_value);

}  // namespace ttnn::operations::data_movement::detail
