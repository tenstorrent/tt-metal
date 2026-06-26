// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/sampling/device/sampling_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct SamplingProgramFactory {
    // Metal 2.0 (MetalV2FactoryConcept). DM-kernel sync-free CBs are handled via the cross-kernel
    // DFB bridge: read-staging k/p (c_14/c_15) relocated reader->writer, and the writer-assembled
    // output CB (c_13) bridged writer-PRODUCER -> compute-CONSUMER (terminal no-op) since a DM
    // kernel cannot self-loop a DFB.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SamplingParams& operation_attributes, const SamplingInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
