// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim::qsr {

struct SliceRmProgramFactory {
    // The src0 DFB's entry_size / num_entries depend on slice_start (via misalignment /
    // unpadded_row_size_bytes), so padded_shape is folded into compute_program_hash() — each
    // unique DFB sizing keeps its own cache entry.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim::qsr
