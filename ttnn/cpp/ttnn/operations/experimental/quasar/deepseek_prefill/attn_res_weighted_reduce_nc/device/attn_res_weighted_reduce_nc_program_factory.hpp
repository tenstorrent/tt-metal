// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"

#include "attn_res_weighted_reduce_nc_device_operation_types.hpp"

namespace ttnn::experimental::prim::qsr {

struct AttnResWeightedReduceNCProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const AttnResWeightedReduceNCParams& operation_attributes,
        const AttnResWeightedReduceNCInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim::qsr
