// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceRmShardedProgramFactory {
    // Both DFBs are built on borrowed memory (borrowed_from the input / output TensorParameter);
    // their backing L1 address resolves from the corresponding TensorArgument each dispatch. DFB
    // entry_size / num_entries are NOT re-applied on a cache hit — padded_shape is folded into
    // compute_program_hash() so each unique sizing gets its own cache entry.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    // Cache-hit hook: the reader args depend only on shapes / slice_start / shard specs, all
    // cache-keyed, so the only per-dispatch state is the two borrowed-memory backing addresses,
    // which travel as tensor bindings.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
