// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Metal 2.0 (MetalV2FactoryConcept) factory for the resnet-shaped multi-core, ROW_MAJOR-layout pad
// path.  The work is split across cores by split_across_cores() (currently hardcoded for resnet
// shapes); each core streams its slice of unpadded input rows into cb_in0, filling padding from the
// op-owned pad-value const tensor (tensor::pad), and the writer drains cb_in0 to the output.
//
// The op-owned pad-value const tensor is allocated once on cache miss inside create_program_artifacts
// and parked on ProgramArtifacts::op_owned_tensors so it outlives the cached Program at a stable
// address (see #44565).
struct PadRmReaderWriterMultiCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);
};
}  // namespace ttnn::prim::qsr
