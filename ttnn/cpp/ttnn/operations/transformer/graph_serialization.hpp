// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/graph/graph_type_serialization.hpp"
#include "ttnn/graph/graph_registration.hpp"

namespace ttnn::operations::transformer {

TTNN_DEFINE_REFLECTION_SERIALIZATION(SDPAProgramConfig);

}  // namespace ttnn::operations::transformer

// Automatic type registration
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::transformer::SDPAProgramConfig);
