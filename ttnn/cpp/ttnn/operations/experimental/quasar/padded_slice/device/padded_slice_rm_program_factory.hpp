// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "padded_slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

// Metal-2.0 (Quasar) row-major padded_slice factory. Quasar cannot build a legacy DataMovementKernel
// (kernel.hpp:382), so this emits a ProgramSpec + QuasarDataMovementKernel: a reader that produces the
// sliced sticks into the (borrowed) sharded output DFB, and a writer that drains/commits it. Mirrors
// the quasar interleaved_to_sharded factory. RM-only; the pad-row path (output_row > input_row) and the
// non-aligned TRID path are ported but not yet numerically validated (the resnet stem is aligned/non-pad).
struct PaddedSliceRMProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim::qsr
