// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unary_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim::qsr {

// Metal 2.0 (ProgramSpec / DataflowBuffer) port of the TILE path of ttnn::operations::unary's
// UnaryDeviceOperation::ProgramFactory: one reader DM thread, one compute (Neo) thread and one writer DM
// thread per node, the output tiles split across the worker grid.
struct UnaryProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim::qsr
