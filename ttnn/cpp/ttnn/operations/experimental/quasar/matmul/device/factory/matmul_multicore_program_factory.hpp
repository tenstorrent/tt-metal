// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim::qsr {

struct MatmulMultiCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::qsr::MatmulParams& operation_attributes,
        const ttnn::prim::qsr::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
