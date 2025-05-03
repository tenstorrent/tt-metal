// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat {

tt::tt_metal::operation::ProgramWithCallbacks rm_repeat_program_factory(
    const Tensor& input, uint32_t num_repeats, const Tensor& output, bool is_last_dim);

}  // namespace ttnn::operations::data_movement::repeat
