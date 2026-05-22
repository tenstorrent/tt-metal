// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "deepseek_moe_post_combine_tilize_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEPostCombineTilizeProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  The tilize-output CB is
    // sharded onto the output tensor buffer (bound via .buffer for dynamic CB
    // address re-application); per-core reader runtime args include the input
    // buffer address and intra-row / row offsets that vary by shard position.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DeepseekMoEPostCombineTilizeParams& operation_attributes,
        const DeepseekMoEPostCombineTilizeInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
