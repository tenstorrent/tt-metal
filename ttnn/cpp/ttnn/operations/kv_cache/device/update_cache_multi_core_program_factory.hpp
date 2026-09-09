// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "update_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

// Metal 2.0 spec factory: builds a ProgramSpec + ProgramRunArgs declaratively. The framework owns
// program construction and caching; the custom concept re-applies per-dispatch state on cache hit
// via override_runtime_arguments (see below).
struct UpdateCacheMultiCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& tensor_return_value);

    // CustomProgramSpecFactoryConcept cache-hit hook: returns the per-dispatch ProgramRunArgs the
    // framework applies via UpdateProgramRunArgs. Re-applies only the hash-excluded, attribute-derived
    // scalars (cache_start_id, Wbytes, tile_update_offset, batch_read_offset) plus the tensor bindings
    // (whose addresses refresh through the typed channel); shape-derived args stay at their cache-miss
    // values.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const KvCacheParams& operation_attributes,
        const KvCacheInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Runtime-arg values that derive from operation_attributes (update_idx, batch_offset) which
// UpdateKVCacheOperation::compute_program_hash deliberately EXCLUDES from the program-cache key
// (so two updates that differ only in those cache-hit), yet are baked into reader/writer runtime
// args. Single source of truth for the work-split + formulas: create_program_artifacts (cache miss)
// and override_runtime_arguments (cache hit) both call it, so the core order and the arg values
// cannot drift between the two.
//   - cache_start_ids: per-core, in the SAME core order create_program_artifacts emits runtime args.
//   - tile_update_offset / batch_read_offset: identical on every core (op-wide scalars).
//   - Wbytes: op-wide; not attribute-derived, but it depends on fp32_dest_acc_en, which the hash
//     excludes along with the rest of compute_kernel_config, so the override re-applies it too.
struct UpdateCacheDynamicArgs {
    std::vector<std::pair<tt::tt_metal::CoreCoord, std::uint32_t>> cache_start_ids;
    std::uint32_t tile_update_offset = 0;
    std::uint32_t batch_read_offset = 0;
    std::uint32_t Wbytes = 0;
};

UpdateCacheDynamicArgs compute_update_cache_dynamic_args(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args);

}  // namespace ttnn::prim
