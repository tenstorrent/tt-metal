// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "fused_recurrent_gated_delta_rule_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct FusedRecurrentGatedDeltaRuleProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FusedRecurrentGatedDeltaRuleParams& attrs,
        const FusedRecurrentGatedDeltaRuleInputs& in,
        std::vector<Tensor>& outputs);
};

}  // namespace ttnn::prim
