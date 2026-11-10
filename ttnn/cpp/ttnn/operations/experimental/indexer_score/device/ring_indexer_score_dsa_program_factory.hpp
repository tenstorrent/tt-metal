// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/per_device_program_merge.hpp"
#include <array>
#include <optional>
#include <vector>

#include <tt-metalium/workload_descriptor.hpp>

#include "indexer_score_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/distributed/types.hpp"  // MeshCoordinate(RangeSet)

namespace ttnn::operations::experimental::indexer_score::program {

// Ring-fused indexer_score factory (descriptor model). Selected only when operation_attributes.fused_ring is
// set; the classic IndexerScoreProgramFactory stays byte-identical for all unfused usage. Mirrors
// RingJointSDPAMeshWorkloadFactory: one ProgramDescriptor per coordinate co-scheduling the indexer compute +
// the ring_attention all-gather (the only Linear+fuse-capable AG) into ONE program, with a producer->consumer
// semaphore handshake. The reader gates each K band on ONLY the SP shards that band's tiles land in (fine-
// grained per-band overlap; NOT a coarse whole-gather barrier), so already-arrived shards score while farther
// slabs are still in flight. DSA-only (relu, num_groups == 1, no block-pool): the fused path serves the
// lightning indexer's all-gather-then-score flow.
namespace detail {
struct RingIndexerScoreDsaDescriptorAdapterOperation {
    using operation_attributes_t = indexer_score::operation_attributes_t;
    using tensor_args_t = indexer_score::tensor_args_t;
    using spec_return_value_t = indexer_score::spec_return_value_t;
    using tensor_return_value_t = indexer_score::tensor_return_value_t;
};
}  // namespace detail

struct RingIndexerScoreDsaProgramFactory {
    // A range's per-chip programs differ only in runtime-arg values: they run as one program with per-device runtime
    // args. The causal geometry words (writer and compute common args) are per device even where the chips agree when
    // the program is built, because RingIndexerScoreDsaMeshWorkloadFactory::override_runtime_arguments sets them per
    // device on a cache hit.
    static constexpr bool merge_per_device_programs = true;
    // A ring end has no fabric connection towards the missing neighbour: its all-gather writer args are shorter. The
    // writer parses them in order behind a has-neighbour flag, so zero padding is never read.
    static constexpr bool pad_per_device_runtime_args = true;
    static std::vector<ttnn::PerDeviceRuntimeArgsWords> per_device_runtime_args_words();
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& args,
        const tensor_args_t& tensors,
        tensor_return_value_t& out,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct RingIndexerScoreDsaMeshWorkloadFactory {
    using descriptor_adapter_t =
        ttnn::device_operation::MeshDeviceOperationAdapter<detail::RingIndexerScoreDsaDescriptorAdapterOperation>::
            DescriptorMeshWorkloadAdapter<RingIndexerScoreDsaProgramFactory>;
    struct AgArgumentPlan {
        uint32_t kernel_idx{};
        // Core placement is structural; runtime storage is still resolved on every update.
        std::vector<tt::tt_metal::CoreCoord> active_cores;
    };
    struct ScalarControls {
        uint32_t chunk_start;
        uint32_t kv_len;
        uint32_t cache_batch;
        uint32_t num_layers;
        uint32_t layer_index;
        uint32_t backward_semaphore;
        uint32_t forward_semaphore;

        bool operator==(const ScalarControls&) const = default;
    };
    struct shared_variables_t {
        typename descriptor_adapter_t::shared_variables_t descriptor;
        std::array<AgArgumentPlan, 4> ag_plans;
        // Per coordinate of the program's range, in its iteration order.
        std::vector<uint32_t> device_indices;
        std::vector<uint32_t> tp_indices;
        // The causal geometry last set per device (chunk start, straddle q, straddle jump per coordinate).
        std::vector<uint32_t> geometry;
        // Invocation-wide guard, stored on the first shared entry. Addresses are
        // always rebound; only unchanged scalar controls can bypass their updates.
        std::optional<ScalarControls> scalar_controls;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensors,
        tensor_return_value_t& out);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const operation_attributes_t& args,
        const tensor_args_t& tensors,
        tensor_return_value_t& out);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<RingIndexerScoreDsaMeshWorkloadFactory>);

}  // namespace ttnn::operations::experimental::indexer_score::program
