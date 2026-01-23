// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/graph/graph_type_serialization.hpp"
#include "ttnn/graph/graph_registration.hpp"

namespace ttnn {

TTNN_DEFINE_REFLECTION_SERIALIZATION(TileReshapeMapMode);

}  // namespace ttnn

// Automatic type registration
TTNN_REGISTER_GRAPH_ARG(ttnn::TileReshapeMapMode);
