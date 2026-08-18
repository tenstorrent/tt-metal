// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "reshard_device_operation_types.hpp"

namespace ttnn::prim {

// Factory for DRAM->DRAM nd reshard (simple page by page copy)
struct NdReshardCopyPagesFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshardParams& operation_attributes, const ReshardInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
