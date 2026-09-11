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

struct SliceRmShardedProgramFactory {
    // Both tensors reach the kernel as borrowed-memory DFBs rather than through accessors: the
    // input and output shards already live in L1, and the reader gathers between them by NoC.
    // The DFB entry sizes vary with shard shape / element size, so padded_shape is folded into
    // compute_program_hash() to keep each unique sizing in its own cache entry. DFB sizing is not
    // re-applied on a cache hit; the cached program carries it.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    // CustomProgramSpecFactoryConcept cache-hit hook. Every reader argument is shape-derived and
    // hashed, so only the two borrowed backing addresses move — and those re-resolve from the
    // tensor bindings supplied here.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
