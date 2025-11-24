// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/program.hpp>
#include <vector>
#include <tuple>
#include <optional>

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program {

struct ReduceScatterProgramArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> all_cores;
    uint32_t num_directions_per_link;
    uint32_t num_workers_per_direction;
    uint32_t num_mux_cores_per_direction_per_link;
    uint32_t num_cores_per_link;
    uint32_t num_links;
};

struct mesh_runtime_params_t {
    const MeshCoordinate sender_device_coord;
    const std::optional<MeshCoordinate> forward_coord;
    const std::optional<MeshCoordinate> backward_coord;
    uint32_t ring_index;
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler;
    std::optional<uint32_t> num_workers_per_direction_opt;
    CoreCoord core_grid_offset;
};

void append_fabric_mux_connection_ct_args(
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    uint32_t num_workers_per_direction,
    std::vector<uint32_t>& writer_ct_args);

void append_fabric_mux_connection_rt_args(
    const bool mux_connection_valid,
    const CoreCoord& mux_virtual_core,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    const CoreCoord& worker_logical_core,
    const uint32_t worker_per_direction_id,
    const bool is_termination_master,
    const CoreCoord termination_master_virtual_core,
    tt::tt_metal::Program& program,
    std::vector<uint32_t>& worker_rt_args);

uint32_t reduce_scatter_minimal_async_core_count_per_link(
    uint32_t num_workers_per_direction,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link);

uint32_t default_chunks_per_sync(
    ttnn::ccl::Topology topology, uint32_t num_tiles_to_process_per_slice, uint32_t tile_granularity);

uint32_t default_workers(
    const distributed::MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    ttnn::ccl::Topology topology,
    uint32_t input_data_size_bytes,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link);

std::tuple<uint32_t, uint32_t, uint32_t> map_nd_to_4d(const ttnn::Shape& shape, const uint32_t dim);

std::tuple<uint32_t, uint32_t, uint32_t> map_2d_to_4d(const uint32_t dim);

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_tile_offsets(
    const uint32_t worker_id,
    const uint32_t num_workers,
    const uint32_t output_batch_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t slice_Wt,
    const uint32_t input_tensor_Wt,
    const uint32_t normalized_dim);

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program
