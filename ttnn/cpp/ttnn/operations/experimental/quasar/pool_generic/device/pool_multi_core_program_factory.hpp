// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/api/ttnn/metal_v2_artifacts.hpp"
#include "pool_op.hpp"

namespace ttnn::operations::pool::quasar {

// Metal 2.0 program factory for Pool2D (max / avg pool, with optional indices).
//
// Single-program MetalV2FactoryConcept factory. The legacy WorkloadDescriptor path
// was the resource-workaround unwind: every per-coord program was structurally
// identical and the two helper tensors (sliding-window reader indices, and the
// avg-pool scalar config) are op-owned tensors carried in ProgramArtifacts.
ttnn::device_operation::ProgramArtifacts pool2d_create_program_artifacts(
    const Pool2D::operation_attributes_t& op_attr,
    const Pool2D::tensor_args_t& tensor_args,
    Pool2D::tensor_return_value_t& output_tensors);

}  // namespace ttnn::operations::pool::quasar
