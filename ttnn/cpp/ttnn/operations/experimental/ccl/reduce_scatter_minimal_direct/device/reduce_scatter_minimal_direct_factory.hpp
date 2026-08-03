// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reduce_scatter_minimal_direct_op_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/workload_descriptor.hpp>

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

// Single source of truth for the geometry above, shared by `compute_output_specs` (which sizes staging
// from it) and the program factory (which wires it into the kernels), so the two can never disagree.
ReduceScatterDirectGeometry reduce_scatter_direct_geometry(
    const ReduceScatterMinimalDirectParams& args, const ttnn::Tensor& input_tensor, bool fp32_dest_acc_en);

// Which fabric configurations this op can run on.
//
// Every send is a multi-hop unicast to one of the two ring directions, over a
// RoutingPlaneConnectionManager built as FabricApiType::Linear. The requirement is that the ACTIVE AXIS
// be a TORUS: it wraps, so every destination is reachable by travelling one direction along a single
// fabric dimension, with no turn (a plain 2D mesh resolves to Topology::Mesh and is rejected here).
//
// The mesh's OTHER extent is irrelevant -- cluster_axis already confines the collective to one axis, so a
// 2x4 with a torus X axis is four independent 4-rings, each exactly the degenerate case above.
//
// What this cannot express, and what the factory rejects with a throw rather than a hang: a logical 1xN
// VIEW that snakes across a larger physical grid (this 2x4 box opened as 1x8 maps to chips 0,1,2,3,7,6,5,4).
// There the "ring" mixes X and Y hops, and 2D routing -- which is by DESTINATION NODE, not hop count --
// cannot be made to follow it. See the factory's direction fix-up for why that matters.
bool reduce_scatter_direct_fabric_supported(
    const ttnn::MeshDevice& mesh_device,
    tt::tt_fabric::FabricConfig fabric_config,
    tt::tt_fabric::Topology axis_topology);

// Resolves the collective's axis the same way the factory does: the caller's cluster_axis if given,
// else the last axis with more than one device.
uint32_t reduce_scatter_direct_active_axis(const ReduceScatterMinimalDirectParams& args);

// Worker-core selection: one core per link, each owning that link's fwd + bwd connection. Factored out
// so the resolved link count and the core placement are defined in exactly one place.
uint32_t reduce_scatter_direct_num_links(const ReduceScatterMinimalDirectParams& args, uint32_t chunks_per_slice);
std::pair<CoreRangeSet, std::vector<CoreCoord>> reduce_scatter_direct_worker_cores(
    const ReduceScatterMinimalDirectParams& args, ttnn::MeshDevice* mesh_device, uint32_t chunks_per_slice);

// Direct (one-shot) reduce-scatter program factory. Declarative workload-scoped form: the framework
// builds and caches the MeshWorkload from the descriptor we return. Returns two tensors -- [0] output
// slice, [1] staging for the incoming contributions.
//
// create_workload_descriptor() runs ONCE per workload (cache miss):
//   1. Allocates the GlobalSemaphores and parks them on WorkloadDescriptor::semaphores, which keeps them
//      alive for the cached workload's lifetime -- every kernel takes their addresses as runtime args.
//      Order is fixed and read back by index: [0 .. num_devices-1] the per-SOURCE arrival counters, then
//      reader_gen, writer_gen, compute_gen, init_sync. See semaphore_index below.
//        - arrival counters are indexed by SOURCE device, so each has exactly one sender and an absolute
//          wait on it cannot be satisfied by a different (raced-ahead) device. Never reset.
//        - the three *_gen counters are per-kernel private invocation counters, each bumped once per
//          launch by its owner, so all three read the same invocation index with no cross-kernel
//          handshake -- that index drives the staging double-buffer parity. (compute's body runs on all
//          three TRISCs, so exactly one of them (PACK) does the increment.)
//        - init_sync is the writer's start barrier, only read when that barrier is compiled in.
//   2. Runs the distributed Synchronize so every device sees the semaphores before any program launches.
//   3. Emits ONE ProgramDescriptor per mesh coordinate: the program depends on the sender coordinate
//      (device_idx, ring neighbours, and the per-destination hop/direction/route table), so the mesh
//      cannot share a single descriptor.
//
// Buffer base addresses are bound as Buffer* through emplace_runtime_args(), and the aliased reduce CB
// via CBDescriptor::buffer, so the framework patches them on the cache-hit fast path -- no manual
// override_runtime_arguments() hook.
struct ReduceScatterMinimalDirectProgramFactory {
    using tensor_return_value_t = std::vector<Tensor>;

    // Layout of WorkloadDescriptor::semaphores. Kept next to the factory so the producer and every
    // consumer index it the same way.
    struct SemaphoreIndex {
        static constexpr size_t arrival_base = 0;  // + source device index
        static size_t reader_gen(uint32_t num_devices) { return num_devices; }
        static size_t writer_gen(uint32_t num_devices) { return num_devices + 1; }
        static size_t compute_gen(uint32_t num_devices) { return num_devices + 2; }
        static size_t init_sync(uint32_t num_devices) { return num_devices + 3; }
        static size_t count(uint32_t num_devices) { return num_devices + 4; }
    };

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ReduceScatterMinimalDirectParams& operation_attributes,
        const ReduceScatterMinimalDirectInputs& tensor_args,
        tensor_return_value_t& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
