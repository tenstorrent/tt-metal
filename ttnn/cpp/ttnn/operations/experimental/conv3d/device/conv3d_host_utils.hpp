// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "conv3d_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Resolve the internal execution policy (use_l1_prefetch, slide_axis) for the
// given Conv3dParams + input shape. Pure function: must produce identical output
// for the same hashed Conv3dParams so program-cache lookup is safe.
//
// Called once before launch (so the resolved policy lands in Conv3dParams and
// participates in the program hash) and re-called inside the program factory
// purely as a consistency check.
Conv3dExecutionPolicy resolve_conv3d_execution_policy(
    const Conv3dParams& params, const ttnn::Shape& input_shape, tt::tt_metal::DataType input_dtype, bool has_bias);

}  // namespace ttnn::experimental::prim
