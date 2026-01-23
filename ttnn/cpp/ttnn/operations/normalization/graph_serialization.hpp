// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_operation_types.hpp"
#include "ttnn/graph/graph_type_serialization.hpp"
#include "ttnn/graph/graph_registration.hpp"

namespace ttnn::operations::normalization {

// LayerNorm serialization
TTNN_DEFINE_REFLECTION_SERIALIZATION(LayerNormDefaultProgramConfig);
TTNN_DEFINE_REFLECTION_SERIALIZATION(LayerNormShardedMultiCoreProgramConfig);

inline std::ostream& operator<<(std::ostream& os, const LayerNormProgramConfig& value) {
    std::visit([&os](const auto& v) { os << v; }, value);
    return os;
}

// Softmax serialization
TTNN_DEFINE_REFLECTION_SERIALIZATION(SoftmaxDefaultProgramConfig);
TTNN_DEFINE_REFLECTION_SERIALIZATION(SoftmaxShardedMultiCoreProgramConfig);

inline std::ostream& operator<<(std::ostream& os, const SoftmaxProgramConfig& value) {
    std::visit([&os](const auto& v) { os << v; }, value);
    return os;
}

}  // namespace ttnn::operations::normalization

// Automatic type registration
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::normalization::LayerNormDefaultProgramConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::normalization::LayerNormShardedMultiCoreProgramConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::normalization::LayerNormProgramConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::normalization::SoftmaxDefaultProgramConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::normalization::SoftmaxShardedMultiCoreProgramConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::normalization::SoftmaxProgramConfig);
