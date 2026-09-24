// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <utility>
#include <vector>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <cstdint>
#include "ttnn/metal_v2_artifacts.hpp"
#include "update_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

// Metal 2.0 spec factory: builds a ProgramSpec + ProgramRunArgs declaratively. The framework owns
// program construction and caching; the custom concept re-applies per-dispatch state on cache hit
// via override_runtime_arguments (see below).
struct FillCacheMultiCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& tensor_return_value);

    // CustomProgramSpecFactoryConcept cache-hit hook: returns the per-dispatch ProgramRunArgs the
    // framework applies via UpdateProgramRunArgs. Re-applies only the hash-excluded, attribute-derived
    // writer cache_start_id plus the tensor bindings; shape-derived args stay at their cache-miss values.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const KvCacheParams& operation_attributes,
        const KvCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Per-core writer cache_start_id, in the SAME core order create_program_artifacts emits runtime args.
// cache_start_id derives from operation_attributes (batch_idx, update_idx) which
// UpdateKVCacheOperation::compute_program_hash deliberately EXCLUDES from the program-cache key
// (so two fills that differ only in those cache-hit), yet it is baked into a writer runtime arg.
// Single source of truth for the work-split + formula: create_program_artifacts (cache miss) and
// override_runtime_arguments (cache hit) both call it, so the core order and the arg values cannot
// drift between the two.
std::vector<std::pair<tt::tt_metal::CoreCoord, std::uint32_t>> compute_fill_cache_start_ids(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args);

}  // namespace ttnn::prim
