// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
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

}  // namespace ttnn::experimental::ccl
