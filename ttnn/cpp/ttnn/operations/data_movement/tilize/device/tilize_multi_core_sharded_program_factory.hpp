// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Sharded factory for HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED inputs.
// Supports two output paths selected at runtime:
//   - Same-layout sharded L1 output: zero-copy (output DFB borrowed from the shard buffer).
//   - L1 INTERLEAVED output: zero-copy input read, TensorAccessor scatter write.
//     Only valid for HEIGHT_SHARDED with ROW_MAJOR orientation (contiguous output tile ranges).
//     DRAM interleaved output is excluded — DRAM writes always require NoC, so the default factory
//     is used instead (no performance benefit from the optimized path).
struct TilizeMultiCoreShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const TilizeParams& operation_attributes,
        const TilizeInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
