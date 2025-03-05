// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::data_movement {

tt::tt_metal::operation::ProgramWithCallbacks bcast_sharded_h_optimised(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& output_tensor,
    ttnn::BcastOpMath bcast_op);

}  // namespace ttnn::operations::data_movement
