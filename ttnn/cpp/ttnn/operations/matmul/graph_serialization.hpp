// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/graph/graph_type_serialization.hpp"
#include "ttnn/graph/graph_registration.hpp"
#include <variant>

namespace ttnn::operations::matmul {

// Define serialization for matmul program config types
TTNN_DEFINE_REFLECTION_SERIALIZATION(MatmulMultiCoreProgramConfig);
TTNN_DEFINE_REFLECTION_SERIALIZATION(MatmulMultiCoreReuseProgramConfig);
TTNN_DEFINE_REFLECTION_SERIALIZATION(MatmulMultiCoreReuseMultiCastProgramConfig);
TTNN_DEFINE_REFLECTION_SERIALIZATION(MatmulMultiCoreReuseMultiCast1DProgramConfig);
TTNN_DEFINE_REFLECTION_SERIALIZATION(MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig);

// Variant serialization
inline std::ostream& operator<<(
    std::ostream& os,
    const std::variant<
        MatmulMultiCoreProgramConfig,
        MatmulMultiCoreReuseProgramConfig,
        MatmulMultiCoreReuseMultiCastProgramConfig,
        MatmulMultiCoreReuseMultiCast1DProgramConfig,
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>& value) {
    std::visit([&os](const auto& v) { os << v; }, value);
    return os;
}

}  // namespace ttnn::operations::matmul

// ═══════════════════════════════════════════════════════════════════════════════
// Automatic type registration via static initialization
// These registrations are queued and executed during GraphArgumentSerializer::initialize()
// No manual function calls needed - completely automatic and scalable!
// ═══════════════════════════════════════════════════════════════════════════════

TTNN_REGISTER_GRAPH_ARG(ttnn::operations::matmul::MatmulMultiCoreProgramConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig);
TTNN_REGISTER_GRAPH_ARG(std::variant<
                        ttnn::operations::matmul::MatmulMultiCoreProgramConfig,
                        ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig,
                        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig,
                        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig,
                        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>);
