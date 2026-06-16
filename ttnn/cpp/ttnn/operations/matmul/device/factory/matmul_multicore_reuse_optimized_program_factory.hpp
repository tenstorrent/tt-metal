// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::prim {

struct MatmulMultiCoreReuseOptimizedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static CoreRangeSet default_core_range(IDevice* device);
};

}  // namespace ttnn::prim
