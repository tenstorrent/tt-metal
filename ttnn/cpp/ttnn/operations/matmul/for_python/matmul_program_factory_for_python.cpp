// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/for_python/matmul_program_factory_for_python.hpp"

#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

namespace ttnn::for_python {

MatmulProgramFactory select_matmul_program_factory(
    const ttnn::prim::MatmulParams& operation_attributes, const ttnn::prim::MatmulInputs& /*tensor_args*/) {
    const auto& config = operation_attributes.program_config.value();

    return std::visit(
        [](const auto& c) -> MatmulProgramFactory {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreProgramConfig>) {
                return MatmulMultiCoreProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                return MatmulMultiCoreReuseOptimizedProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return MatmulMultiCoreReuseMcast2DProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(!c.gather_in0, "gather_in0 is not supported in the descriptor interface");
                return MatmulMultiCoreReuseMcast1DProgramFactory{};
            } else if constexpr (std::is_same_v<
                                     T,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                return MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory{};
            } else if constexpr (
                std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
                return MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory{};
            } else {
                TT_THROW("Unknown program config type");
            }
        },
        config);
}

}  // namespace ttnn::for_python
