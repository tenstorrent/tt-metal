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

// Returns the default chunks_per_sync value for the given topology and tile counts.
uint32_t reduce_scatter_default_chunks_per_sync(
    ttnn::ccl::Topology topology, uint32_t num_tiles_to_process_per_slice, uint32_t tile_granularity);

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
std::optional<ttnn::TensorSpec> reduce_scatter_ring_interm_staging_spec(
    const ttnn::Tensor& input_tensor,
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
