// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::data_movement {

operation::ProgramWithCallbacks bcast_sharded_h(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const Tensor &output_tensor,
    ttnn::BcastOpMath bcast_op);

} // ttnn::operations::data_movement
