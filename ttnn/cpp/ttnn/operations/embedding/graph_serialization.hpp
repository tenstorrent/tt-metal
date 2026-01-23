// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/embedding/device/embedding_device_operation_types.hpp"
#include "ttnn/graph/graph_type_serialization.hpp"
#include "ttnn/graph/graph_registration.hpp"

namespace ttnn::operations::embedding {

TTNN_DEFINE_REFLECTION_SERIALIZATION(EmbeddingsType);

}  // namespace ttnn::operations::embedding

// Automatic type registration
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::embedding::EmbeddingsType);
