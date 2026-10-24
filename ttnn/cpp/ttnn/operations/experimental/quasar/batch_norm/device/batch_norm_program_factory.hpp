// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "batch_norm_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Interleaved inference batch_norm (Metal 2.0 / MetalV2FactoryConcept). The unit of work is one (n, c)
// plane of Ht x Wt tiles. 4 reader harts / 4 NEOs / 2 writer harts with DFB implicit sync (see the
// factory's thread plan).
struct BatchNormProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const BatchNormParams& operation_attributes, const BatchNormInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
