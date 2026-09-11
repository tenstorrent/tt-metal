// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceTileProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    // CustomProgramSpecFactoryConcept cache-hit hook. The per-core scalars below are excluded from
    // compute_program_hash, so a divergent-partition cache hit would otherwise leave them stale and
    // produce an all-zero output (#52651); they are re-applied on every hit, together with the tensor
    // bindings the framework re-points addresses through.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// One node's reader/writer scalar arguments for the two TILE factories.
struct SliceTilePerNodeArgs {
    tt::tt_metal::CoreCoord node;
    // False for a node the work split left with nothing to do. Such a node is still given a full
    // argument row so every node's layout is identical.
    bool active = false;
    uint32_t start_id = 0;
    uint32_t num_tiles = 0;
    // num_dims entries; the reader's runtime vararg block, seeded here and advanced on device.
    std::vector<uint32_t> id_per_dim;
    // Tiles emitted by all preceding nodes — the writer's start_id on an active node.
    uint32_t num_tiles_written = 0;
};

struct SliceTileWorkSplit {
    tt::tt_metal::CoreRangeSet all_nodes;
    std::vector<SliceTilePerNodeArgs> per_node;
};

// Single source of truth for the TILE factories' work split and per-node scalars, shared by
// create_program_artifacts (cache miss) and override_runtime_arguments (cache hit) so the node order
// and the values cannot drift between the two. `start_offset` is the tile-index base the reader adds
// to every node's start_id: SliceTile passes get_tiled_start_offset(...), SliceTileTensorArgs passes
// 0 because it computes the real offset on device from the start tensor.
SliceTileWorkSplit slice_tile_work_split(
    const SliceParams& args, const SliceInputs& tensor_args, const Tensor& output, uint32_t start_offset);

}  // namespace ttnn::prim
