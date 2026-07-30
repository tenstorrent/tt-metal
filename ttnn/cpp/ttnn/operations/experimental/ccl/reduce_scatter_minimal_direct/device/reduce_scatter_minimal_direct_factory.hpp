// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reduce_scatter_minimal_direct_op_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

#include <utility>
#include <vector>

namespace ttnn::experimental::prim {

// Page-space geometry of the scatter, for ANY scatter dim and ANY rank.
//
// The tiled input buffer is a row-major array of pages over [d0, ..., d_{rank-3}, Ht, Wt]. Slicing it on
// dim `d` cuts a middle axis of that array, so slice j of every "outer" index (the product of the dims
// before d) is a CONTIGUOUS run of `slice_run_pages` pages starting at j * slice_run_pages, and
// consecutive runs sit `stride_pages` apart. That is the only thing the scatter dim changes: the reader
// walks (run, stride) instead of a hardcoded (slice_Wt, width_tiles) pair, and everything downstream --
// chunking, staging, the output write -- only ever sees the resulting linear page order of a slice.
// The last dim is the degenerate case with stride == one row and run == the slice's width in tiles;
// dim 0 is the other degenerate case with a single run (outer == 1), so the stride is never taken.
//
// This is also why the op needs no ND -> canonical-4D collapse (the (B, C, Ht, Wt) + normalized_dim
// mapping the ring ops carry, whose kernels are a fixed 4-level nested loop): the pair above already IS
// that collapse. Everything below the scatter dim folds into `inner_pages` and everything above folds
// into the outer count, for any rank >= 2 -- rank never reaches a kernel.
struct ReduceScatterDirectGeometry {
    uint32_t single_tile_bytes;        // bytes per tile
    uint32_t tile_granularity;         // tiles per chunk (compute/CB granularity)
    uint32_t interm_tiles_per_packet;  // max tiles carried in one fabric packet
    uint32_t page_bytes;               // staging row width == tile_granularity * single_tile_bytes
    uint32_t pages_per_slice;          // tiles in one slice (== output tiles)
    uint32_t chunks_per_slice;         // ceil(pages_per_slice / tile_granularity)
    uint32_t slice_run_pages;          // contiguous pages of a slice, per outer index
    uint32_t stride_pages;             // input pages between the starts of consecutive runs
};

// Single source of truth for the geometry above, shared by compute_output_specs (which sizes staging
// from it) and the program factory (which wires it into the kernels), so the two can never disagree.
ReduceScatterDirectGeometry reduce_scatter_direct_geometry(
    const ReduceScatterMinimalDirectParams& args, const ttnn::Tensor& input_tensor, bool fp32_dest_acc_en);

// Worker-core selection: one core per link, each owning that link's fwd + bwd connection. Factored out
// so the resolved link count and the core placement are defined in exactly one place.
uint32_t reduce_scatter_direct_num_links(const ReduceScatterMinimalDirectParams& args, uint32_t chunks_per_slice);
std::pair<CoreRangeSet, std::vector<CoreCoord>> reduce_scatter_direct_worker_cores(
    const ReduceScatterMinimalDirectParams& args, ttnn::MeshDevice* mesh_device, uint32_t chunks_per_slice);

// Direct (one-shot) reduce-scatter program factory: one MeshWorkload per participating device
// coordinate via create_at, cached and rebound by override_runtime_arguments. Returns two tensors --
// [0] output slice, [1] staging for the incoming contributions.
struct ReduceScatterMinimalDirectMeshWorkloadFactory {
    struct shared_variables_t {
        // One worker core per link (each owns that link's fwd + bwd connection).
        std::vector<tt::tt_metal::CoreCoord> worker_cores;
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle compute_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        // Arrival counters indexed by SOURCE device: source s increments arrival_sems[s] on every peer's
        // mirror core, so each counter has exactly one sender and an absolute wait on it cannot be
        // satisfied by a different (raced-ahead) device. Never reset; waits target base + 1 where base is
        // the reader's private invocation counter.
        std::vector<tt::tt_metal::GlobalSemaphore> arrival_sems;
        // Per-core private invocation counters (one per kernel). Each is bumped once per program launch by
        // its owner, so all three always read the same invocation index without any cross-kernel
        // handshake -- that index drives the staging double-buffer parity. The compute one is only read on
        // the sharded path (where compute must pick a parity half itself); its body runs on all three
        // TRISCs, so exactly one of them (PACK) does the increment.
        tt::tt_metal::GlobalSemaphore reader_gen_sem;
        tt::tt_metal::GlobalSemaphore writer_gen_sem;
        tt::tt_metal::GlobalSemaphore compute_gen_sem;
        // Writer start-barrier counter: every peer's mirror core increments it once per invocation, and
        // the writer holds its sends until all N-1 have. Only read when the writer's init sync is
        // compiled in (op-allocated buffers); held here so its address stays valid either way.
        tt::tt_metal::GlobalSemaphore init_sync_sem;
        // Set only on the sharded path, where cb_reduce is globally allocated on top of the staging
        // tensor and so has to be rebound whenever that buffer moves.
        std::optional<tt::tt_metal::CBHandle> reduce_cb_handle;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    using tensor_return_value_t = std::vector<Tensor>;

    static cached_mesh_workload_t create_mesh_workload(
        const ReduceScatterMinimalDirectParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ReduceScatterMinimalDirectInputs& tensor_args,
        tensor_return_value_t& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ReduceScatterMinimalDirectParams& operation_attributes,
        const ReduceScatterMinimalDirectInputs& tensor_args,
        tensor_return_value_t& output_tensors);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const ReduceScatterMinimalDirectParams& operation_attributes,
        const ttnn::MeshCoordinate& sender_device_coord,
        const ReduceScatterMinimalDirectInputs& tensor_args,
        const tensor_return_value_t& output_tensors,
        const std::vector<tt::tt_metal::GlobalSemaphore>& arrival_sems,
        const tt::tt_metal::GlobalSemaphore& reader_gen_sem,
        const tt::tt_metal::GlobalSemaphore& writer_gen_sem,
        const tt::tt_metal::GlobalSemaphore& compute_gen_sem,
        const tt::tt_metal::GlobalSemaphore& init_sync_sem,
        bool needs_init_sync);
};

}  // namespace ttnn::experimental::prim
