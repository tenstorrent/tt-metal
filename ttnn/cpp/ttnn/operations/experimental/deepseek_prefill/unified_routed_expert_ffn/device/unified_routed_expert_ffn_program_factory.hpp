// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "unified_routed_expert_ffn_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

struct UnifiedRoutedExpertFfnProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UnifiedRoutedExpertFfnParams& operation_attributes,
        const UnifiedRoutedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value);

    // TILE in-place calls store x and output as the same buffer twice in tensor_args, so
    // resolve_bindings bails and the adapter would rebuild this factory on every cache hit.
    // Common runtime args are the only per-dispatch addresses; per-core args stay hashed.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const UnifiedRoutedExpertFfnParams& operation_attributes,
        const UnifiedRoutedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
