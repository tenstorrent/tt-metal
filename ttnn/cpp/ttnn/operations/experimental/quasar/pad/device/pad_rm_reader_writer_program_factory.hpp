// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Metal 2.0 (MetalV2FactoryConcept) factory for the single-core, ROW_MAJOR-layout pad path.
//
// The pad-value const tensor is an op-owned device tensor: allocated and filled once on cache
// miss inside create_program_artifacts() and returned on ProgramArtifacts::op_owned_tensors so
// the framework parks it at a stable address for the cached Program's life (re-patched, never
// re-allocated, on cache hits).  The reader kernel reaches it through a TensorBinding (tensor::pad)
// rather than the legacy buffer-address runtime arg.
struct PadRmReaderWriterProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim::qsr
