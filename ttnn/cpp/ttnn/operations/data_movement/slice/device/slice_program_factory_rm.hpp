// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include <optional>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceRmProgramFactory {
    // The input DFB's entry_size / num_entries depend on slice_start (via misalignment /
    // unpadded_row_size_bytes), so padded_shape is folded into compute_program_hash() — each
    // unique DFB sizing keeps its own cache entry.  DFB sizing is set once at spec construction
    // and is not re-applied on a cache hit.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    // Cache-hit hook: only the two tensor bindings move; every other argument is derived from
    // values the cache key already separates on.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
