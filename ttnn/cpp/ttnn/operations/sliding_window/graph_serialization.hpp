// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
#include "ttnn/graph/graph_type_serialization.hpp"
#include "ttnn/graph/graph_registration.hpp"

namespace ttnn::operations::op_slicing {

TTNN_DEFINE_REFLECTION_SERIALIZATION(Op2DSliceConfig);

}  // namespace ttnn::operations::op_slicing

// Automatic type registration
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::op_slicing::Op2DSliceConfig);
