// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "chunk_gated_delta_rule_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct ChunkGatedDeltaRuleProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkGatedDeltaRuleParams& attrs, const ChunkGatedDeltaRuleInputs& in, std::vector<Tensor>& outputs);
};

}  // namespace ttnn::prim
