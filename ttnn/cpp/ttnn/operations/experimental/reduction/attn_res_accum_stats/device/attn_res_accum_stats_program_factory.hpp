// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include <tt-metalium/program_descriptors.hpp>

#include "attn_res_accum_stats_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct AttnResAccumStatsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnResAccumStatsParams& operation_attributes,
        const AttnResAccumStatsInputs& tensor_args,
        std::array<Tensor, 2>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
