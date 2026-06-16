// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::prim {

struct Conv2dShardedProgramFactory {
    // Metal 2.0 program factory (ProgramSpecFactoryConcept). Produces the immutable
    // ProgramSpec, its mutable ProgramRunArgs, and the op-owned conv_reader_indices
    // tensor (host-populated index table backing the READER_INDICES borrowed-memory
    // DFB). Ports the dense / height-sharded L1 config-tensor path only:
    //   - config_tensors_in_dram == false (the in-DRAM path threads a raw buffer
    //     address through a CTA -> Metal 2.0 framework blocker; TT_FATALs here).
    //   - HEIGHT_SHARDED, non-1D-depthwise (BLOCK_SHARDED / depthwise / the
    //     input_cores != output_cores noop-core path TT_FATAL here; they remain on
    //     the legacy concept and would be ported separately).
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
