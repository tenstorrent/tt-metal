// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_program_utils.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/experimental/ccl/composite_common.hpp"

namespace ttnn::experimental::ccl {

uint32_t reduce_scatter_core_count_per_link(
    uint32_t num_workers_per_direction,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link) {
    log_trace(
        tt::LogOp,
        "DEBUG: num_workers_per_direction: {}, num_directions_per_link: {}, num_mux_cores_per_direction_per_link: {}",
        num_workers_per_direction,
        num_directions_per_link,
        num_mux_cores_per_direction_per_link);
    return num_directions_per_link * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
}

uint32_t reduce_scatter_default_workers(
    const ttnn::MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    ttnn::ccl::Topology topology,
    uint32_t input_data_size_bytes,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link) {
    auto sd_id = sub_device_id.value_or(mesh_device.get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    uint32_t num_cores = subdevice_core_range_set.num_cores();
    log_trace(tt::LogOp, "DEBUG: num_cores: {}", num_cores);
    ttnn::SmallVector<uint32_t> candidate_worker_counts;
    double data_moved_per_link_bytes = double(input_data_size_bytes) * (ring_size - 1) / ring_size / num_links /
                                       (topology == ttnn::ccl::Topology::Ring ? 2 : 1);
    log_trace(tt::LogOp, "DEBUG: data_moved_per_link_bytes: {}", data_moved_per_link_bytes);
    // Heuristic thresholds derived from sweep tests:
    // tests/ttnn/multidevice_perf_tests/test_reduce_scatter_hyperparameter_sweep_perf_galaxy.py
    // For linear: 4+MB → 8 workers; 0.5–4MB → 4 workers; 0–0.5MB → 2 workers.
    // For ring:  50+MB → 8 workers;   1–50MB → 4 workers;   0–1MB → 2 workers.
    // At a single packet size (4KB) use one worker to minimise mux overhead.
    constexpr double RING_HIGH_DATA_THRESHOLD = 50.0 * 1024 * 1024;
    constexpr double RING_LOW_DATA_THRESHOLD = 1.0 * 1024 * 1024;
    constexpr double LINEAR_HIGH_DATA_THRESHOLD = 4000000.0;
    constexpr double LINEAR_LOW_DATA_THRESHOLD = 500000.0;
    constexpr double SINGLE_PACKET_THRESHOLD = 4.0 * 1024;
    if (topology == ttnn::ccl::Topology::Ring) {
        if (data_moved_per_link_bytes > RING_HIGH_DATA_THRESHOLD) {
            candidate_worker_counts = {8, 4, 2, 1};
        } else if (data_moved_per_link_bytes <= SINGLE_PACKET_THRESHOLD) {
            candidate_worker_counts = {1};
        } else if (data_moved_per_link_bytes < RING_LOW_DATA_THRESHOLD) {
            candidate_worker_counts = {2, 1};
        } else {
            candidate_worker_counts = {4, 2, 1};
        }
    } else if (topology == ttnn::ccl::Topology::Linear) {
        if (data_moved_per_link_bytes > LINEAR_HIGH_DATA_THRESHOLD) {
            candidate_worker_counts = {8, 4, 2, 1};
        } else if (data_moved_per_link_bytes <= SINGLE_PACKET_THRESHOLD) {
            candidate_worker_counts = {1};
        } else if (data_moved_per_link_bytes < LINEAR_LOW_DATA_THRESHOLD) {
            candidate_worker_counts = {2, 1};
        } else {
            candidate_worker_counts = {4, 2, 1};
        }
    }
    for (auto worker_count : candidate_worker_counts) {
        uint32_t core_count =
            num_links * reduce_scatter_core_count_per_link(
                            worker_count, num_directions_per_link, num_mux_cores_per_direction_per_link);
        log_trace(tt::LogOp, "DEBUG: core_count {} for worker_count {}", core_count, worker_count);
        if (num_cores >= core_count) {
            log_trace(
                tt::LogOp,
                "data_moved_per_link_bytes: {} and worker_count: {}",
                data_moved_per_link_bytes,
                worker_count);
            return worker_count;
        }
    }
    TT_THROW(
        "Not enough cores available on the subdevice or device for the requested configuration to match the number of "
        "links {}",
        num_links);
}

uint32_t reduce_scatter_default_chunks_per_sync(
    ttnn::ccl::Topology topology, uint32_t num_tiles_to_process_per_slice, uint32_t tile_granularity) {
    // For Line, as early as 20 chunks per sync we get statistically significant performance improvements.
    // For Ring there is no statistically significant performance improvement until 80 chunks per sync.
    TT_FATAL(topology == ttnn::ccl::Topology::Ring || topology == ttnn::ccl::Topology::Linear, "Invalid topology");
    constexpr uint32_t RING_DEFAULT_CHUNKS_PER_SYNC = 80;
    constexpr uint32_t LINEAR_DEFAULT_CHUNKS_PER_SYNC = 20;
    uint32_t default_value =
        topology == ttnn::ccl::Topology::Ring ? RING_DEFAULT_CHUNKS_PER_SYNC : LINEAR_DEFAULT_CHUNKS_PER_SYNC;
    uint32_t total_chunks = std::max(num_tiles_to_process_per_slice / tile_granularity / 2, (uint32_t)1);
    return std::min(default_value, total_chunks);
}

std::tuple<uint32_t, uint32_t, uint32_t> reduce_scatter_map_nd_to_4d(const ttnn::Shape& shape, uint32_t dim) {
    TT_FATAL(shape.rank() > 2, "Expected rank 3 or greater");

    auto [normalized_dim, rank_diff] = composite_common::normalize_dim_4d(dim, shape.rank());

    const uint32_t c_dims_end = shape.rank() - 2;
    uint32_t b_dims_end;
    if (rank_diff >= 1 && dim <= rank_diff) {
        b_dims_end = dim;
        normalized_dim = 1;
    } else if (rank_diff == -1 && dim == 0) {
        b_dims_end = 1;
    } else {
        b_dims_end = shape.rank() - 3;
    }

    const uint32_t input_tensor_B =
        std::accumulate(shape.cbegin(), shape.cbegin() + b_dims_end, 1, std::multiplies<uint32_t>());
    const uint32_t input_tensor_C =
        std::accumulate(shape.cbegin() + b_dims_end, shape.cbegin() + c_dims_end, 1, std::multiplies<uint32_t>());

    return {normalized_dim, input_tensor_C, input_tensor_B};
}

std::tuple<uint32_t, uint32_t, uint32_t> reduce_scatter_map_2d_to_4d(uint32_t dim) {
    constexpr auto RANK_2D = 2;
    TT_FATAL(dim == 0 || dim == 1, "Expected dim 0 or 1");
    const uint32_t normalized_dim = std::get<0>(composite_common::normalize_dim_4d(dim, RANK_2D));
    return {normalized_dim, /*input_tensor_C=*/1, /*input_tensor_B=*/1};
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> reduce_scatter_get_tile_offsets(
    uint32_t worker_id,
    uint32_t num_workers,
    uint32_t output_batch_num_pages,
    uint32_t output_channel_num_pages,
    uint32_t slice_Wt,
    uint32_t input_tensor_Wt,
    uint32_t normalized_dim) {
    uint32_t start_tiles_read;
    uint32_t start_tiles_to_read;
    uint32_t start_pages_read_in_row;
    uint32_t start_row_offset;

    if (normalized_dim == 0) {
        start_tiles_read = worker_id * output_batch_num_pages / num_workers;
        start_tiles_to_read = (worker_id + 1) * output_batch_num_pages / num_workers;
        start_pages_read_in_row = 0;
        start_row_offset = 0;
    } else {
        start_tiles_read = worker_id * output_channel_num_pages / num_workers;
        start_tiles_to_read = (worker_id + 1) * output_channel_num_pages / num_workers;
        start_pages_read_in_row = start_tiles_read % slice_Wt;
        start_row_offset = start_tiles_read / slice_Wt * input_tensor_Wt;
    }

    return {start_tiles_read, start_tiles_to_read, start_pages_read_in_row, start_row_offset};
}

void append_fabric_mux_connection_ct_args(
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    uint32_t num_workers_per_direction,
    std::vector<uint32_t>& writer_ct_args) {
    constexpr auto num_ct_args = 5;
    const std::array<uint32_t, num_ct_args> ct_args = {
        mux_kernel_config.get_num_buffers(channel_type),
        mux_kernel_config.get_buffer_size_bytes(channel_type),
        mux_kernel_config.get_status_address(),
        mux_kernel_config.get_termination_signal_address(),
        num_workers_per_direction};
    writer_ct_args.reserve(writer_ct_args.capacity() + num_ct_args);
    std::copy(ct_args.begin(), ct_args.end(), std::back_inserter(writer_ct_args));
}

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
    std::vector<uint32_t>& worker_rt_args) {
    constexpr auto num_rt_args = 17;
    const std::array<uint32_t, num_rt_args> rt_args = {
        mux_connection_valid,
        is_termination_master,
        mux_virtual_core.x,
        mux_virtual_core.y,
        mux_kernel_config.get_channel_base_address(channel_type, worker_per_direction_id),
        mux_kernel_config.get_connection_info_address(channel_type, worker_per_direction_id),
        mux_kernel_config.get_connection_handshake_address(channel_type, worker_per_direction_id),
        mux_kernel_config.get_flow_control_address(channel_type, worker_per_direction_id),
        mux_kernel_config.get_buffer_index_address(channel_type, worker_per_direction_id),
        mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_per_direction_id),
        tt::tt_metal::CreateSemaphore(program, {worker_logical_core}, 0),
        tt::tt_metal::CreateSemaphore(program, {worker_logical_core}, 0),
        tt::tt_metal::CreateSemaphore(program, {worker_logical_core}, 0),
        tt::tt_metal::CreateSemaphore(program, {worker_logical_core}, 0),
        tt::tt_metal::CreateSemaphore(program, {worker_logical_core}, 0),
        termination_master_virtual_core.x,
        termination_master_virtual_core.y,
    };
    worker_rt_args.reserve(worker_rt_args.capacity() + num_rt_args);
    std::copy(rt_args.begin(), rt_args.end(), std::back_inserter(worker_rt_args));
}

}  // namespace ttnn::experimental::ccl
