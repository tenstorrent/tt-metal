// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "practice_routed_expert_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert {

struct PracticeRoutedExpertProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PracticeRoutedExpertParams& operation_attributes,
        const PracticeRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert
