// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unified_routed_expert_ffn_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

struct UnifiedRoutedExpertFfnProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UnifiedRoutedExpertFfnParams& operation_attributes,
        const UnifiedRoutedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
