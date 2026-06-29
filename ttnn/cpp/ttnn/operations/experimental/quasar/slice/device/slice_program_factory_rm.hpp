// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/program_spec_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Metal 2.0 (ProgramSpecFactoryConcept) factory for the interleaved + B/W-sharded
// row-major no-step slice path.
//
// The src0 DataflowBuffer's entry_size / num_entries depend on slice_start (via
// misalignment / unpadded_row_size_bytes), so each unique slice layout produces a
// distinct ProgramSpec and therefore its own cache entry (the spec is the cache key).
struct SliceRmProgramFactory {
    static ttnn::device_operation::ProgramSpecArtifacts create_program_spec(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim::qsr
