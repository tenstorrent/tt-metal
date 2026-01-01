// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "softmax_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Interleaved last-dim softmax (Metal 2.0 / MetalV2FactoryConcept).
// 4 reader harts / 4 NEOs / 2 writer harts with DFB implicit sync (see the factory's thread plan).
struct SoftmaxProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SoftmaxParams& operation_attributes, const SoftmaxInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
