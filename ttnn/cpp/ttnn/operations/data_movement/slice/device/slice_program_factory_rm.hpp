// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceRmProgramFactory {
    // The src0 DFB's entry_size / num_entries depend on slice_start (via misalignment /
    // unpadded_row_size_bytes), so padded_shape is folded into compute_program_hash() — each unique
    // sizing keeps its own cache entry. DFB sizing is not re-applied on a cache hit; the cached
    // program carries it.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    // CustomProgramSpecFactoryConcept cache-hit hook. Every scalar is shape-derived and hashed, so
    // only the tensor bindings move — but this concept applies nothing on its own, so they are
    // re-supplied here.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
