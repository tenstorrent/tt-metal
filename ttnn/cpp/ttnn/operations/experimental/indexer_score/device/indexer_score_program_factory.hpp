// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/workload/legacy/program_descriptors.hpp>

#include "indexer_score_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"  // MeshCoordinate

namespace ttnn::operations::experimental::indexer_score::program {

// Classic (unfused) factory. One program per mesh coordinate: each device derives its own chunk_start and
// straddle terms from its coordinate. A single device is a 1x1 mesh and builds one program at index 0.
struct IndexerScoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& args,
        const tensor_args_t& tensors,
        tensor_return_value_t& out,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Cache hit: re-applies the buffer addresses and the hash-excluded values (cache_batch_idx, kv_len,
    // chunk_start_idx) for this coordinate. Everything else is fixed by the program hash.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& args,
        const tensor_args_t& tensors,
        tensor_return_value_t& out,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::experimental::indexer_score::program
