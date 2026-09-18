// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_sp_async/device/matmul_reduce_scatter_sp_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_fusion_common.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"

namespace ttnn::experimental::prim {

struct MatmulReduceScatterSpAsyncSharedVariables {
    ReduceScatterProgramArtifacts reduce_scatter_artifacts;
    ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t matmul_shared_variables;
};

struct MatmulReduceScatterSpAsyncProgramFactory {
    using shared_variables_t = MatmulReduceScatterSpAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MatmulReduceScatterSpAsyncParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MatmulReduceScatterSpAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MatmulReduceScatterSpAsyncParams& args,
        const MatmulReduceScatterSpAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensors);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const MatmulReduceScatterSpAsyncParams& args,
        const ttnn::MeshCoordinate& mesh_coord,
        const MatmulReduceScatterSpAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensors);
};

// ---- Core grid split -------------------------------------------------------------------------------------------
// The CCL workers take the bottom `ccl_core_rows` rows (choose_worker_cores walks the grid row-major from the
// offset), the matmul the rows above, anchored at (0,0).
tt::tt_metal::CoreCoord sp_matmul_core_grid(const Tensor& input, uint32_t ccl_core_rows);
tt::tt_metal::CoreCoord sp_reduce_scatter_core_grid_offset(const Tensor& input, uint32_t ccl_core_rows);
// Worker + mux cores the RS builders will ask for (mirrors their per-topology mux rule).
uint32_t sp_reduce_scatter_core_count(ttnn::ccl::Topology topology, uint32_t num_links, uint32_t num_workers_per_link);
// Default reduce-scatter workers per direction per link: the measured best on this grid (5 on Ring, 4 on Linear;
// see the perf table in test_matmul_reduce_scatter_sp_async.py), halved until the worker+mux cores fit the
// reserved rows.
uint32_t sp_default_reduce_scatter_workers(
    const Tensor& input, ttnn::ccl::Topology topology, uint32_t num_links, uint32_t ccl_core_rows);

// MatmulParams with bcast_batch=true and the program config filled in (user override or the derived per-sub-batch
// config on the [B*T,1,S/T,K] view from sp_matmul_fusion_common::sub_batched_view).
ttnn::prim::MatmulParams resolve_sp_matmul_params(
    const MatmulReduceScatterSpAsyncParams& args, const MatmulReduceScatterSpAsyncInputs& tensor_args);

// ---- Schedule plumbing -----------------------------------------------------------------------------------------
// The matmul sub-batch order for rank `ring_index`: batch-major, and within a batch the order in which that rank's
// reduce-scatter readers first touch their local slices (rs_first_touch_order), so the collective can start on the
// first finished slice. Entry j: in0_idx == out_idx == b*T + s (the [B*T,1,S/T,X] views share the sub-batch index).
std::vector<ttnn::experimental::ccl::SpSubBatch> sp_matmul_schedule(
    ttnn::ccl::Topology topology, uint32_t B, uint32_t T, uint32_t ring_index);

// The order in which the reduce-scatter readers of rank `r` first read their LOCAL input slices, derived from the
// reader kernels:
//   Ring   (ring_reduce_scatter_minimal_async_reader.cpp): both direction cores start at slice r+T/2 and walk T/2+1
//          iterations, the forward core decreasing (r+T/2-1, ...) and the backward core increasing (r+T/2+1, ...),
//          both ending on the local slice r. First-touch order: r+T/2, {r+T/2-1, r+T/2+1}, {r+T/2-2, r+T/2+2}, ..., r.
//   Linear (line_reduce_scatter_minimal_async_reader.cpp): the FWD core reads T-1, T-2, ..., r+1 then r (final
//          reduction), the BWD core reads 0, 1, ..., r-1 then r. Interleaved FWD/BWD, r last.
// Returns a permutation of 0..T-1 whose last element is r.
std::vector<uint32_t> rs_first_touch_order(ttnn::ccl::Topology topology, uint32_t T, uint32_t r);

}  // namespace ttnn::experimental::prim
