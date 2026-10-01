// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "rebank_rm_merge_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RebankRmMergeFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RebankRmMergeParams& operation_attributes,
        const RebankRmMergeTensorArgs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
