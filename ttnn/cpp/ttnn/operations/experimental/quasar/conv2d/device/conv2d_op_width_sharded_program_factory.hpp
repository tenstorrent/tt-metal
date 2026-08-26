// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"

namespace ttnn::prim::qsr {

struct Conv2dWidthShardedProgramFactory {
    // Metal 2.0 factory entry point.  Builds the immutable ProgramSpec + mutable
    // ProgramRunArgs (paired in ProgramArtifacts) for the width-sharded conv2d.
    //
    // The intermediate conv_reader_indices tensor — which must outlive the cached
    // program — is allocated here and parked on ProgramArtifacts::op_owned_tensors
    // (the adapter keeps it alive in the program cache so its device-memory
    // allocation stays at a stable address across dispatches).  This replaces the
    // legacy WorkloadDescriptor::buffers parking.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
