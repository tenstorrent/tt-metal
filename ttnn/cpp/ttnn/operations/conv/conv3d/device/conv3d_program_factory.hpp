// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::conv::conv3d::detail {

operation::ProgramWithCallbacks conv3d_factory(
    const Tensor& input_tensor, const Conv3dConfig& config, const Tensor& output_tensor);

}  // namespace ttnn::operations::conv::conv3d::detail
