// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GatherCodegenParams;
struct GatherCodegenInputs;

// Host-computed tile-page geometry, computed once from tensor shapes and copied into
// GatherCodegenParams so it participates in program-cache hashing; compute_gather_geometry itself is
// called only at the gather_codegen() call site, never re-derived inside the factories or
// select_program_factory.
struct GatherGeometry {
    uint32_t Ht = 0;
    uint32_t Wt_input = 0;
    uint32_t Wt_index = 0;
    uint32_t index_valid_h_last = 0;
    uint32_t index_valid_w_last = 0;
    uint32_t index_ht_per_batch = 0;
};

GatherGeometry compute_gather_geometry(const Tensor& input_tensor, const Tensor& input_index_tensor);

// Output tiles the row-buffered writers batch into one NoC write burst before syncing
// (kernels/gather_writer.cpp, gather_writer_tiled.cpp).
constexpr uint32_t kGatherWriteBatchTiles = 4;

// Depth of the row-buffered plans' output CB, in output tile pages. A row shorter than one write
// batch still has to hold a whole batch, or the writer's wait_front() for it can never be satisfied.
// The writers clamp each burst against this same depth to keep a flat multi-tile read off the ring
// wrap, so they take it as a compile-time arg rather than recomputing it: a writer that disagreed
// with the CB it pops would clamp to the wrong wrap point.
constexpr uint32_t gather_output_cb_tiles(uint32_t Wt_index) {
    return Wt_index > kGatherWriteBatchTiles ? Wt_index : kGatherWriteBatchTiles;
}

// Whether the row-buffered kernel's three CBs (Wt_input + 1 + gather_output_cb_tiles(Wt_index) tile
// pages, the SAME depths the Interleaved/Tiled factories allocate) fit the device's real per-core L1
// budget.
bool gather_interleaved_fits_l1(
    const Tensor& input_tensor, const Tensor& input_index_tensor, uint32_t Wt_input, uint32_t Wt_index);

// Whether the SHALLOWEST plan any gather factory can be built with fits per-core L1. Every other
// plan scales down to this one (streaming's input CB bottoms out at two pages), so a call this
// rejects has no feasible codegen dispatch at all and the routing gate must send it to native.
//
// A pure function of static tensor and device properties: unlike gather_interleaved_fits_l1 and
// gather_streaming_chunk_tiles, which the program factory consults, this one is a ROUTING gate and
// must not read live L1 occupancy -- ttnn::gather()'s router and the prim's validate evaluate it at
// different points of the same dispatch and are consistent only while the answer cannot move.
bool gather_min_plan_fits_l1(const Tensor& input_tensor, const Tensor& input_index_tensor);

// Depth of the streaming factory's input CB, in input tile pages. The streaming reader rescans the
// whole index tile once per resident block of input tiles, so its scalar cost per output tile is
// ceil(Wt_input / depth) * TILE_HW: a block deep enough to hold the entire row costs the single scan
// the row-buffered reader pays. The block COUNT is therefore what the L1 budget buys (the deepest
// block the L1 left over by the fixed index and output pages affords, capped at the row); the depth
// returned here is the row spread evenly over that count, which keeps the scan count while dropping
// the tail-block padding re-reads the writer would otherwise issue.
uint32_t gather_streaming_chunk_tiles(const Tensor& input_tensor, const Tensor& input_index_tensor, uint32_t Wt_input);

// Row-buffered: full Wt_input row resident in L1 (kernels/gather_reader.cpp, gather_writer.cpp).
struct GatherCodegenProgramFactoryInterleaved {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor);
};

// Per-output-tile split: high parallelism for small Ht (kernels/gather_reader_tiled.cpp,
// gather_writer_tiled.cpp).
struct GatherCodegenProgramFactoryTiled {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor);
};

// Chunked streaming: large Wt_input that doesn't fit the row-buffered L1 budget, walked in
// gather_streaming_chunk_tiles()-deep blocks (kernels/gather_reader_streaming.cpp,
// gather_writer_streaming.cpp).
struct GatherCodegenProgramFactoryStreaming {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
