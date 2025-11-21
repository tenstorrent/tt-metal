// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program_common.hpp"
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_device_operation.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"

#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {

namespace operations::experimental::ccl::detail {

uint32_t reduce_scatter_minimal_async_core_count_per_link(
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

uint32_t default_workers(
    const MeshDevice& mesh_device,
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
    // Heuristic values are based on the sweep test:
    // tests/ttnn/multidevice_perf_tests/test_reduce_scatter_hyperparameter_sweep_perf_galaxy.py
    if (topology == ttnn::ccl::Topology::Ring) {
        // For ring, 50+MB is where 8 workers start scaling. 1-50MB is where 4 workers start scaling. 0-1MB is where 2
        // workers start scaling.
        constexpr double RING_HIGH_DATA_THRESHOLD_MB = 50.0;
        constexpr double RING_LOW_DATA_THRESHOLD_MB = 1.0;
        if (data_moved_per_link_bytes > RING_HIGH_DATA_THRESHOLD_MB) {
            candidate_worker_counts = {8, 4, 2, 1};
        } else if (data_moved_per_link_bytes < RING_LOW_DATA_THRESHOLD_MB) {
            candidate_worker_counts = {2, 1};
        } else {
            candidate_worker_counts = {4, 2, 1};
        }
    } else if (topology == ttnn::ccl::Topology::Linear) {
        // For linear, 4+MB is where 8 workers start scaling. 0.5-4MB is where 4 workers start scaling. 0-0.5MB is where
        // 2 workers start scaling.
        constexpr double LINEAR_HIGH_DATA_THRESHOLD_MB = 4.0;
        constexpr double LINEAR_LOW_DATA_THRESHOLD_MB = 0.5;
        if (data_moved_per_link_bytes > LINEAR_HIGH_DATA_THRESHOLD_MB) {
            candidate_worker_counts = {8, 4, 2, 1};
        } else if (data_moved_per_link_bytes < LINEAR_LOW_DATA_THRESHOLD_MB) {
            candidate_worker_counts = {2, 1};
        } else {
            candidate_worker_counts = {4, 2, 1};
        }
    }
    for (auto worker_count : candidate_worker_counts) {
        uint32_t core_count =
            num_links * reduce_scatter_minimal_async_core_count_per_link(
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
        "Not enough cores available on the subdevice or device for the requested match the number of links {}",
        num_links);
}

uint32_t default_chunks_per_sync(
    ttnn::ccl::Topology topology, uint32_t num_tiles_to_process_per_slice, uint32_t tile_granularity) {
    // For Line, as early as 20 chunks per sync we get statistically significant performance improvements
    // For Ring there is no statistically significant performance improvements until 80 chunks per sync, which beats the
    // default value of syncing once This was determined by the sweep test:
    // tests/ttnn/multidevice_perf_tests/test_reduce_scatter_hyperparameter_sweep_perf_galaxy.py
    TT_FATAL(topology == ttnn::ccl::Topology::Ring || topology == ttnn::ccl::Topology::Linear, "Invalid topology");
    constexpr uint32_t RING_DEFAULT_CHUNKS_PER_SYNC = 80;
    constexpr uint32_t LINEAR_DEFAULT_CHUNKS_PER_SYNC = 20;
    uint32_t default_value =
        topology == ttnn::ccl::Topology::Ring ? RING_DEFAULT_CHUNKS_PER_SYNC : LINEAR_DEFAULT_CHUNKS_PER_SYNC;
    uint32_t total_chunks = std::max(num_tiles_to_process_per_slice / tile_granularity / 2, (uint32_t)1);
    return std::min(default_value, total_chunks);
}

std::tuple<uint32_t, uint32_t, uint32_t> map_nd_to_4d(const ttnn::Shape& shape, const uint32_t dim) {
    // Here we do a couple of tricks so that the kernels can handle ND tensors
    // implicitly reshape lower dims so it is treated as 4D

    TT_FATAL(shape.rank() > 2, "Expected rank 3 or greater");

    auto [normalized_dim, rank_diff] = composite_common::normalize_dim_4d(dim, shape.rank());

    const uint32_t c_dims_end = shape.rank() - 2;
    uint32_t b_dims_end;
    if (rank_diff >= 1 && dim <= rank_diff) {
        // gather dim to rank-3 accumulated into C
        b_dims_end = dim;
        normalized_dim = 1;
    } else if (rank_diff == -1 && dim == 0) {
        // scattering on dim 0 of rank 3 tensor sets normalized_dim to 0
        // need special case to set b_dims_end accordingly
        b_dims_end = 1;
    } else {
        // C will be 4D normalized dim 1
        b_dims_end = shape.rank() - 3;
    }

    const uint32_t input_tensor_B =
        std::accumulate(shape.cbegin(), shape.cbegin() + b_dims_end, 1, std::multiplies<uint32_t>());

    const uint32_t input_tensor_C =
        std::accumulate(shape.cbegin() + b_dims_end, shape.cbegin() + c_dims_end, 1, std::multiplies<uint32_t>());

    return std::make_tuple(normalized_dim, input_tensor_C, input_tensor_B);
};

std::tuple<uint32_t, uint32_t, uint32_t> map_2d_to_4d(const uint32_t dim) {
    constexpr auto RANK_2D = 2;
    TT_FATAL(dim == 0 || dim == 1, "Expected dim 0 or 1");

    const uint32_t normalized_dim = std::get<0>(composite_common::normalize_dim_4d(dim, RANK_2D));
    const uint32_t input_tensor_C = 1, input_tensor_B = 1;

    return std::make_tuple(normalized_dim, input_tensor_C, input_tensor_B);
};

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_tile_offsets(
    const uint32_t worker_id,
    const uint32_t num_workers,
    const uint32_t output_batch_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t slice_Wt,
    const uint32_t input_tensor_Wt,
    const uint32_t normalized_dim) {
    uint32_t start_tiles_read;
    uint32_t start_tiles_to_read;
    uint32_t start_pages_read_in_row;
    uint32_t start_row_offset;

    if (normalized_dim == 0) {
        start_tiles_read = worker_id * output_batch_num_pages / num_workers;
        start_tiles_to_read = (worker_id + 1) * output_batch_num_pages / num_workers;

        start_pages_read_in_row = 0;  // not used for dim 0 scatter
        start_row_offset = 0;         // not used for dim 0 scatter
    } else {
        start_tiles_read = worker_id * output_channel_num_pages / num_workers;
        start_tiles_to_read = (worker_id + 1) * output_channel_num_pages / num_workers;

        start_pages_read_in_row = start_tiles_read % slice_Wt;
        start_row_offset = start_tiles_read / slice_Wt * input_tensor_Wt;
    }

    return std::make_tuple(start_tiles_read, start_tiles_to_read, start_pages_read_in_row, start_row_offset);
}

}  // namespace operations::experimental::ccl::detail

}  // namespace ttnn

namespace ttnn {
namespace ccl {

void append_fabric_mux_connection_ct_args(
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    uint32_t num_workers_per_direction,
    std::vector<uint32_t>& writer_ct_args) {
    constexpr auto num_ct_args = 5;
    const std::array<uint32_t, num_ct_args> ct_args = {
        mux_kernel_config.get_num_buffers(channel_type),        // fabric_mux_num_buffers_per_channel
        mux_kernel_config.get_buffer_size_bytes(channel_type),  // fabric_mux_channel_buffer_size_bytes
        mux_kernel_config.get_status_address(),                 // fabric_mux_status_address
        mux_kernel_config.get_termination_signal_address(),     // fabric_mux_termination_signal_address
        num_workers_per_direction                               // num_mux_clients
    };

    writer_ct_args.reserve(writer_ct_args.capacity() + num_ct_args);
    std::copy(ct_args.begin(), ct_args.end(), std::back_inserter(writer_ct_args));
}

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
    std::vector<uint32_t>& worker_rt_args) {
    constexpr auto num_rt_args = 17;
    const std::array<uint32_t, num_rt_args> rt_args = {
        mux_connection_valid,   // mux_connection_valid
        is_termination_master,  // is_termination_master
        mux_virtual_core.x,     // fabric_mux_x
        mux_virtual_core.y,     // fabric_mux_y
        mux_kernel_config.get_channel_base_address(
            channel_type, worker_per_direction_id),  // fabric_mux_channel_base_address
        mux_kernel_config.get_connection_info_address(
            channel_type, worker_per_direction_id),  // fabric_mux_connection_info_address
        mux_kernel_config.get_connection_handshake_address(
            channel_type, worker_per_direction_id),  // fabric_mux_connection_handshake_address
        mux_kernel_config.get_flow_control_address(
            channel_type, worker_per_direction_id),  // fabric_mux_flow_control_address
        mux_kernel_config.get_buffer_index_address(
            channel_type, worker_per_direction_id),  // fabric_mux_buffer_index_address
        mux_kernel_config.get_channel_credits_stream_id(
            channel_type, worker_per_direction_id),          // fabric_mux_channel_id
        CreateSemaphore(program, {worker_logical_core}, 0),  // termination_sync_address
        CreateSemaphore(program, {worker_logical_core}, 0),  // local_fabric_mux_status_address
        CreateSemaphore(program, {worker_logical_core}, 0),  // local_flow_control_address
        CreateSemaphore(program, {worker_logical_core}, 0),  // local_teardown_address
        CreateSemaphore(program, {worker_logical_core}, 0),  // local_buffer_index_address
        termination_master_virtual_core.x,                   // termination_master_noc_x
        termination_master_virtual_core.y,                   // termination_master_noc_y
    };

    worker_rt_args.reserve(worker_rt_args.capacity() + num_rt_args);
    std::copy(rt_args.begin(), rt_args.end(), std::back_inserter(worker_rt_args));
}

}  // namespace ccl
}  // namespace ttnn
