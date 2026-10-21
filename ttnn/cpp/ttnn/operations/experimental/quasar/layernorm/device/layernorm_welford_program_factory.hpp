// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "layernorm_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Interleaved, non-distributed Welford layer_norm (Metal 2.0 / MetalV2FactoryConcept).
// 4 reader harts / 4 NEOs / 2 writer harts with DFB implicit sync (see the factory's thread plan).
struct LayerNormWelfordProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const LayerNormParams& operation_attributes, const LayerNormInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
