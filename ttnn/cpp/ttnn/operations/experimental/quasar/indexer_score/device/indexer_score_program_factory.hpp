// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/workload/legacy/program_descriptors.hpp>

#include "indexer_score_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"  // MeshCoordinate

namespace ttnn::operations::experimental::quasar::indexer_score::program {

// Per-coordinate descriptor factory: one program per mesh coordinate so each device derives its own
// chunk_start from its coordinate (the 4-arg create_descriptor makes the framework call it once per coord). A
// single device is a 1x1 mesh, so the single-chip path flows through the same code with one coordinate (index 0).
struct IndexerScoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& args,
        const tensor_args_t& tensors,
        tensor_return_value_t& out,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);

    // Cache hit: re-applies the q/k/weights/out addresses and the hash-excluded cache_batch_idx / kv_len /
    // chunk_start_idx (and the causal geometry derived from chunk_start_idx + this coordinate's SP/TP rank).
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& args,
        const tensor_args_t& tensors,
        tensor_return_value_t& out,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::experimental::quasar::indexer_score::program
