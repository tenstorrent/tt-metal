// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/graph/graph_type_serialization.hpp"
#include "ttnn/graph/graph_registration.hpp"

namespace ttnn::operations::conv::conv2d {

TTNN_DEFINE_REFLECTION_SERIALIZATION(Conv2dConfig);

}  // namespace ttnn::operations::conv::conv2d

// Automatic type registration
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::conv::conv2d::Conv2dConfig);
