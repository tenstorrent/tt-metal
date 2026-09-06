// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::ccl {

// Returns the number of worker + mux cores needed per link.
uint32_t reduce_scatter_core_count_per_link(
    uint32_t num_workers_per_direction,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link);

// Selects the default number of workers per direction based on data size heuristics.
uint32_t reduce_scatter_default_workers(
    const ttnn::MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    ttnn::ccl::Topology topology,
    uint32_t input_data_size_bytes,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link);

// Returns the default chunks_per_sync value for the given topology and chunking geometry.
//
// Takes the per-worker tile range and the repeat count (units per worker for dims 1-3, batches for dim 0)
// SEPARATELY rather than pre-multiplied: the kernels chunk each repeat independently, so the number
// of chunks a step issues is repeats * ceil(tiles / granularity), which is not recoverable from the
// product once the two have been multiplied together.
uint32_t reduce_scatter_default_chunks_per_sync(
    ttnn::ccl::Topology topology,
    uint32_t tiles_per_worker_per_repeat,
    uint32_t num_repeats,
    uint32_t tile_granularity);

// Cap on the default chunks_per_sync for the ring kernels that carry a worker's whole share of the
// slice in every step (scatter dims 1-3). A step there holds up to 48 chunks, so "half the chunks"
// reaches 16-24 and delays the receiver's start on each sync group. Measured on a 1x8 Blackhole ring
// (bf16, 2 links): 4 was never worse than the uncapped default and beat it by 6.5% at 8M elements (16
// chunks per step) and by ~1% at 16M-24M (32-48 chunks per step). An interval of 1 loses 5-21% on
// steps of 8-48 chunks to the per-chunk waits. The dim 0 kernels, which split a step between the two
// directions chunk by chunk, prefer the longer interval by 1.5-2% at 24M and are not capped.
constexpr uint32_t RING_UNIT_STEP_MAX_CHUNKS_PER_SYNC = 4;

// Sizing for the chunk-paged "contiguous" intermediate used by the ring reduce-scatter fast path.
//
// The contiguous path replaces scatter-writes to the intermediate with a single contiguous
// fused-unicast write per fabric packet. To make each chunk's tiles land at contiguous
// destination bytes, the intermediate is laid out as a row-major interleaved-DRAM UINT8 tensor
// whose page (row) holds exactly one chunk (tile_granularity tiles). See
// rs-contiguous-interm-design for the addressing contract.
struct RingIntermStagingParams {
    bool use_contiguous;               // true => allocate/address the chunk-paged staging intermediate
    uint32_t normalized_dim;           // canonical 4D scatter dim
    uint32_t tile_granularity;         // tiles per chunk (compute/CB granularity)
    uint32_t single_tile_bytes;        // bytes per tile
    uint32_t interm_tiles_per_packet;  // max tiles carried in one fabric packet (payload / single_tile_bytes)
    uint32_t chunks_per_channel;       // ceil(output_channel_num_pages / tile_granularity)
    uint32_t total_chunks;             // ring_size * slice_C * chunks_per_channel (== staging num pages)
    uint32_t page_bytes;               // tile_granularity * single_tile_bytes (staging row width, must be DRAM-aligned)
};

// Derives the contiguous-intermediate sizing from the input tensor + op parameters. Shared by
// compute_output_specs (to size the staging tensor) and the ring program factory (to wire kernel
// args) so both agree exactly. page_bytes must be a multiple of the device DRAM alignment (checked
// by the program factory). The contiguous path applies to Ring + dim != 0 regardless of whether the
// intermediate is internally allocated or a caller-provided persistent buffer.
RingIntermStagingParams reduce_scatter_ring_interm_staging_params(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t ring_size,
    bool fp32_dest_acc_en);

// Builds the TensorSpec for the contiguous chunk-paged staging intermediate (row-major UINT8,
// interleaved DRAM, page = one chunk). Returns nullopt when the contiguous path does not apply.
// Single source of truth shared by compute_output_specs and the python-exposed allocation helper, so
// an internally allocated intermediate and a caller-provided persistent buffer are byte-identical.
std::optional<tt::tt_metal::TensorSpec> reduce_scatter_ring_interm_staging_spec(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t ring_size,
    bool fp32_dest_acc_en);

// Builds the TensorSpec for the penult intermediate: a small chunk-paged region (same
// row-major UINT8 / interleaved DRAM layout as the main intermediate) used by the ring contiguous
// path's second-to-last iteration to stage one direction's contribution ahead of schedule, instead
// of scatter-writing it directly into the tiled output tensor. Unlike the main intermediate, this
// buffer is addressed without a slice_idx term (shape total_chunks/ring_size == slice_C *
// chunks_per_channel pages): each device receives exactly one such contribution, from exactly one
// neighbor, at exactly one iteration, so no ring-position axis is needed. Returns nullopt when the
// contiguous path does not apply. See rs-contiguous-interm-design.
std::optional<tt::tt_metal::TensorSpec> reduce_scatter_ring_penult_intermediate_staging_spec(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t ring_size,
    bool fp32_dest_acc_en);

// True when `tensor` is laid out exactly as `spec` describes in every respect the kernels depend on
// (logical shape, dtype, layout, buffer type).
bool reduce_scatter_tensor_matches_spec(const ttnn::Tensor& tensor, const tt::tt_metal::TensorSpec& spec);

// Chooses which intermediate staging layout a given reduce-scatter call uses.
//
// The chunk-paged ("contiguous") layout only exists for Ring + scatter dim != 0. Within that:
//   - no caller-provided intermediate: contiguous. The op allocates the staging buffer itself, so
//     there is no reason to fall back to the slower tiled layout.
//   - caller-provided intermediate: whichever layout its TensorSpec matches. This lets a caller
//     holding an input-shaped persistent buffer keep using the tiled path.
// Returns false for every other configuration (Linear, scatter dim 0, or a caller-provided
// input-shaped intermediate).
//
// An intermediate matching neither layout is rejected by validate_on_program_cache_miss, so by the
// time the program factory calls this the answer is unambiguous.
bool reduce_scatter_use_contiguous_interm(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t ring_size,
    bool fp32_dest_acc_en);

// Maps an ND tensor shape + dim to a canonical 4D (normalized_dim, C, B) representation.
// Requires rank >= 3.
std::tuple<uint32_t, uint32_t, uint32_t> reduce_scatter_map_nd_to_4d(const ttnn::Shape& shape, uint32_t dim);

// Maps a 2D tensor dim to the canonical 4D representation (normalized_dim=2 or 3, C=1, B=1).
std::tuple<uint32_t, uint32_t, uint32_t> reduce_scatter_map_2d_to_4d(uint32_t dim);

// Computes per-worker tile read start/end offsets for the scatter dimension.
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> reduce_scatter_get_tile_offsets(
    uint32_t worker_id,
    uint32_t num_workers,
    uint32_t output_batch_num_pages,
    uint32_t output_channel_num_pages,
    uint32_t slice_Wt,
    uint32_t input_tensor_Wt,
    uint32_t normalized_dim);

// Per-worker share of one ring step, for the dims that iterate over channels (dim 0 has its own kernels).
//
// The ring kernels process every (batch, channel) pair of the slice inside each ring step, so the
// tensor crosses the ring once however many batches it has. A "unit" is one such pair, indexed
// u = b * slice_C + c over U = input_tensor_B * slice_C units, and the split hands each worker a
// contiguous range of them:
//
//   unit-major (unit_start..unit_end a contiguous span, whole pages within)
//       Each worker owns whole channels of whole batches. The per-channel loop is entered
//       U/num_workers times per step and every visit carries a full channel of pages.
//
//   page-major (unit_start=0, unit_end=U)
//       Every worker visits every unit and takes a fraction of the pages inside each. Used when the
//       units do not divide evenly among the workers, or for a single worker.
//
// Either way a worker moves total_slice_pages / num_workers tiles per step, the same share the dim 0
// kernels give their workers, and balance is identical between the two forms.
struct ReduceScatterWorkerSplit {
    uint32_t unit_start;
    uint32_t unit_end;
    uint32_t start_tiles_read;
    uint32_t start_tiles_to_read;
    uint32_t start_pages_read_in_row;
    uint32_t start_row_offset;
};

ReduceScatterWorkerSplit reduce_scatter_get_worker_split(
    uint32_t worker_id,
    uint32_t num_workers,
    uint32_t input_tensor_B,
    uint32_t slice_C,
    uint32_t output_batch_num_pages,
    uint32_t output_channel_num_pages,
    uint32_t slice_Wt,
    uint32_t input_tensor_Wt,
    uint32_t normalized_dim);

// Appends fabric mux compile-time args to writer_ct_args.
void append_fabric_mux_connection_ct_args(
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    uint32_t num_workers_per_direction,
    std::vector<uint32_t>& writer_ct_args);

// Appends fabric mux run-time args (connection info + semaphores) to worker_rt_args.
void append_fabric_mux_connection_rt_args(
    bool mux_connection_valid,
    const tt::tt_metal::CoreCoord& mux_virtual_core,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    const tt::tt_metal::CoreCoord& worker_logical_core,
    uint32_t worker_per_direction_id,
    bool is_termination_master,
    tt::tt_metal::CoreCoord termination_master_virtual_core,
    tt::tt_metal::Program& program,
    std::vector<uint32_t>& worker_rt_args);

// ProgramDescriptor (Contract-2) variant — same wire layout as the legacy helper
// (17 args in the order listed above), but allocates the five worker-side
// semaphores by pushing SemaphoreDescriptors onto desc.semaphores and writes
// the resulting args into a KernelDescriptor::RTArgList so callers can feed
// the list directly into KernelDescriptor::emplace_runtime_args. The legacy
// Program& helper is preserved; consumers migrate one at a time.
void append_fabric_mux_connection_rt_args(
    bool mux_connection_valid,
    const tt::tt_metal::CoreCoord& mux_virtual_core,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    const tt::tt_metal::CoreCoord& worker_logical_core,
    uint32_t worker_per_direction_id,
    bool is_termination_master,
    tt::tt_metal::CoreCoord termination_master_virtual_core,
    tt::tt_metal::ProgramDescriptor& desc,
    tt::tt_metal::KernelDescriptor::RTArgList& worker_rt_args);

}  // namespace ttnn::experimental::ccl
