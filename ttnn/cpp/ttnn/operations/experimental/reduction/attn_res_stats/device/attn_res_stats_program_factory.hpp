// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "attn_res_stats_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct AttnResStatsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnResStatsParams& operation_attributes,
        const AttnResStatsInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
