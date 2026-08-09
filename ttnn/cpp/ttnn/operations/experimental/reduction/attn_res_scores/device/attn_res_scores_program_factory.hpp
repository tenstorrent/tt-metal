// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "attn_res_scores_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct AttnResScoresProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnResScoresParams& operation_attributes,
        const AttnResScoresInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
