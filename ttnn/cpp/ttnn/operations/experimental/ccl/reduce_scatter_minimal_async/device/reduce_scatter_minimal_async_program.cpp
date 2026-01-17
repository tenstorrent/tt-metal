// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
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

// Import types from the new TMP pattern
using ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::detail::ReduceScatterProgramArtifacts;

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
    // For linear, 4+MB is where 8 workers start scaling. 0.5-4MB is where 4 workers start scaling. 0-0.5MB is where
    // 2 workers start scaling.
    // For ring, 50+MB is where 8 workers start scaling. 1-50MB is where 4 workers start scaling. 0-1MB is where 2
    // workers start scaling.
    // At a single packet size (4KB) we should just have one worker with the optimal packet size and save on mux
    // overheads
    constexpr double RING_HIGH_DATA_THRESHOLD = 50.0 * 1024 * 1024;
    constexpr double RING_LOW_DATA_THRESHOLD = 1.0 * 1024 * 1024;
    constexpr double LINEAR_HIGH_DATA_THRESHOLD = 4000000.0;
    constexpr double LINEAR_LOW_DATA_THRESHOLD = 500000.0;
    constexpr double SINGLE_PACKET_THRESHOLD = 4.0 * 1024;
    if (topology == ttnn::ccl::Topology::Ring) {
        constexpr double SINGLE_PACKET_THRESHOLD = 4.0 * 1024;
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

auto map_nd_to_4d(const ttnn::Shape& shape, const uint32_t dim) {
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

auto map_2d_to_4d(const uint32_t dim) {
    constexpr auto RANK_2D = 2;
    TT_FATAL(dim == 0 || dim == 1, "Expected dim 0 or 1");

    const uint32_t normalized_dim = std::get<0>(composite_common::normalize_dim_4d(dim, RANK_2D));
    const uint32_t input_tensor_C = 1, input_tensor_B = 1;

    return std::make_tuple(normalized_dim, input_tensor_C, input_tensor_B);
};

auto get_tile_offsets(
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

std::vector<uint32_t> get_ring_reader_compile_args(
    const uint32_t ring_index,
    const uint32_t ring_size,
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t output_tensor_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const bool fuse_op,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            ring_index,               // my_chip_id
            ring_size,                // ring_size
            input_cb_index,           // cb_input_id
            intermediate_cb_index,    // cb_intermediate_id
            reader_output_cb_index,   // cb_reader_output_id
            tile_granularity,         // tile_granularity
            page_size,                // page_size
            output_tensor_num_pages,  // output_num_pages
            input_batch_num_pages,    // batch_num_pages
            slice_B,                  // slice_B
        };
    }
    return {
        ring_index,               // my_chip_id
        ring_size,                // ring_size
        input_cb_index,           // cb_input_id
        intermediate_cb_index,    // cb_intermediate_id
        reader_output_cb_index,   // cb_reader_output_id
        tile_granularity,         // tile_granularity
        page_size,                // page_size
        input_batch_num_pages,    // input_batch_num_pages
        input_channel_num_pages,  // input_channel_num_pages
        input_tensor_B,           // input_tensor_B
        input_tensor_Wt,          // input_tensor_Wt
        slice_C,                  // slice_C
        slice_Ht,                 // slice_Ht
        slice_Wt,                 // slice_Wt
        fuse_op,                  // fused         op
        normalized_dim,           // dim normalized to 4D
    };
}

std::vector<uint32_t> get_ring_writer_compile_args(
    const uint32_t ring_index,
    const uint32_t ring_size,
    const uint32_t compute_output_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t num_tiles_to_write_per_packet,
    const uint32_t output_tensor_num_pages,
    const uint32_t output_batch_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            ring_index,                     // my_chip_id
            ring_size,                      // ring_size
            compute_output_cb_index,        // cb_compute_output_id
            reader_output_cb_index,         // cb_reader_output_id
            tile_granularity,               // packet_size_in_pages
            page_size,                      // page_size
            num_tiles_to_write_per_packet,  // num_tiles_to_write_per_packet
            output_tensor_num_pages,        // output_num_pages
            input_batch_num_pages,          // batch_num_pages
            slice_B,                        // slice_B
        };
    }
    return {
        ring_index,                     // my_chip_id
        ring_size,                      // ring_size
        compute_output_cb_index,        // cb_compute_output_id
        reader_output_cb_index,         // cb_reader_output_id
        tile_granularity,               // packet_size_in_pages
        page_size,                      // page_size
        num_tiles_to_write_per_packet,  // num_tiles_to_write_per_packet
        output_batch_num_pages,         // output_batch_num_pages
        input_channel_num_pages,        // input_channel_num_pages
        output_channel_num_pages,       // output_channel_num_pages
        input_tensor_B,                 // input_tensor_B
        input_tensor_Wt,                //         input_tensor_Wt
        slice_C,                        // slice_C
        slice_Ht,                       // slice_Ht
        slice_Wt,                       // slice_Wt
        normalized_dim                  // dim normalized to 4D
    };
}

std::vector<uint32_t> get_ring_reduce_compile_args(
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t compute_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t ring_size,
    const uint32_t input_tensor_B,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            input_cb_index,           // input_cb_id
            intermediate_cb_index,    // intermediate_cb
            compute_output_cb_index,  // output_cb
            tile_granularity,         // tile_granularity
            ring_size,                // ring_size
            slice_B,                  // slice_B
        };
    }
    return {
        input_cb_index,           //         input_cb_id
        intermediate_cb_index,    // intermediate_cb
        compute_output_cb_index,  // output_cb
        tile_granularity,         // tile_granularity
        ring_size,                // ring_size
        input_tensor_B,           // input_tensor_B
        slice_C,                  // slice_C
    };
}

std::vector<uint32_t> get_line_reader_compile_args(
    const uint32_t ring_index,
    const uint32_t ring_size,
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t input_tensor_num_pages,
    const uint32_t output_tensor_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t output_batch_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const bool fuse_op,
    const uint32_t sync_with_other_direction,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            ring_index,                // my_chip_id
            ring_size,                 // ring_size
            input_cb_index,            // cb_input_id
            intermediate_cb_index,     // cb_intermediate_id
            reader_output_cb_index,    // cb_reader_output_id
            tile_granularity,          // tile_granularity
            page_size,                 // page_size
            input_tensor_num_pages,    // input_num_pages
            output_tensor_num_pages,   // output_num_pages
            input_batch_num_pages,     // batch_num_pages
            slice_B,                   // slice_B
            sync_with_other_direction  // sync_with_other_direction
        };
    }
    return {
        ring_index,                 // my_chip_id
        ring_size,                  // ring_size
        input_cb_index,             // cb_input_id
        intermediate_cb_index,      // cb_intermediate_id
        reader_output_cb_index,     // cb_reader_output_id
        tile_granularity,           // tile_granularity
        page_size,                  // page_size
        input_tensor_num_pages,     // input_num_pages
        input_batch_num_pages,      // input_batch_num_pages
        input_channel_num_pages,    // input_channel_num_pages
        output_batch_num_pages,     // output_batch_num_pages
        output_channel_num_pages,   // output_channel_num_pages
        input_tensor_B,             // input_tensor_B
        input_tensor_Wt,            //         input_tensor_Wt
        slice_C,                    // slice_C
        slice_Ht,                   // slice_Ht
        slice_Wt,                   // slice_Wt
        fuse_op,                    //         fuse_op
        sync_with_other_direction,  // sync_with_other_direction
        normalized_dim,             // dim
    };
}

std::vector<uint32_t> get_line_writer_compile_args(
    const uint32_t ring_size,
    const uint32_t compute_output_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t tiles_to_write_per_packet,
    const uint32_t input_tensor_num_pages,
    const uint32_t output_tensor_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t output_batch_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const uint32_t normalized_dim,
    const uint32_t sync_with_other_direction) {
    if (normalized_dim == 0) {
        return {
            ring_size,                  // ring_size
            compute_output_cb_index,    // cb_compute_output_id
            reader_output_cb_index,     // cb_reader_output_id
            tile_granularity,           // tile_granularity
            page_size,                  // page_size
            tiles_to_write_per_packet,  // contig_pages_advanced
            input_tensor_num_pages,     // input_num_pages
            output_tensor_num_pages,    // output_num_pages
            input_batch_num_pages,      // batch_num_pages
            slice_B,                    // slice_B
            sync_with_other_direction,  // sync_with_other_direction
        };
    }
    return {
        ring_size,                  // ring_size
        compute_output_cb_index,    // cb_compute_output_id
        reader_output_cb_index,     // cb_reader_output_id
        tile_granularity,           // tile_granularity
        page_size,                  // page_size
        tiles_to_write_per_packet,  //         contig_pages_advanced
        input_tensor_num_pages,     // input_num_pages
        input_batch_num_pages,      // input_batch_num_pages
        input_channel_num_pages,    // input_channel_num_pages
        output_batch_num_pages,     // output_batch_num_pages
        output_channel_num_pages,   // output_channel_num_pages
        input_tensor_B,             //         input_tensor_b
        input_tensor_Wt,            // input_tensor_Wt
        slice_C,                    // slice_C
        slice_Ht,                   // slice_Ht
        slice_Wt,                   //         slice_Wt
        normalized_dim,             // dim
        sync_with_other_direction   // sync_with_other_direction
    };
}

std::vector<uint32_t> get_line_reduce_compile_args(
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t compute_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t input_tensor_B,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {input_cb_index, intermediate_cb_index, compute_output_cb_index, tile_granularity, slice_B};
    }
    return {input_cb_index, intermediate_cb_index, compute_output_cb_index, tile_granularity, input_tensor_B, slice_C};
}

}  // namespace operations::experimental::ccl::detail

using namespace ccl;

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

ReduceScatterProgramArtifacts build_ring_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset) {
    auto* mesh_device = input_tensor.device();
    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;

    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    bool fuse_op = fused_op_signaler.has_value();

    // op hyperparams
    // Get worker cores
    // 2 senders per direction (2: forward, backward) per link (num_links)
    // Each sender is reader + compute + writer
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    uint32_t input_data_size_bytes = input_tensor.buffer()->size();
    uint32_t num_workers_per_direction =
        num_workers_per_direction_opt.value_or(operations::experimental::ccl::detail::default_workers(
            *mesh_device,
            sub_device_id,
            topology,
            input_data_size_bytes,
            num_links,
            ring_size,
            num_directions_per_link,
            num_mux_cores_per_direction_per_link));
    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    uint32_t num_cores_per_link =
        operations::experimental::ccl::detail::reduce_scatter_minimal_async_core_count_per_link(
            num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        topology, sender_device_coord, forward_coord, backward_coord, mesh_device);
    auto [mcast_forward_args, mcast_backward_args] = ccl::get_forward_backward_line_mcast_configuration(
        topology, sender_device_coord, forward_coord, backward_coord, ring_size - 1, ring_size - 1, mesh_device);

    const auto [all_core_range, all_cores] =
        choose_worker_cores(num_links, num_cores_per_link, mesh_device, sub_device_id, core_grid_offset);

    const auto mux_connection_valid = [&backward_coord, &forward_coord](const uint32_t dir) {
        return (!dir && backward_coord.has_value()) || (dir && forward_coord.has_value());
    };

    std::vector<CoreRange> sender_worker_core_ranges;
    std::vector<CoreRange> mux_core_ranges;
    std::vector<CoreRange> termination_master_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (mux_connection_valid(dir)) {
                mux_core_ranges.emplace_back(mux_core);
            }

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                sender_worker_core_ranges.emplace_back(worker_core);

                if (worker == 0) {
                    termination_master_core_ranges.emplace_back(worker_core);
                }
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_core_ranges);

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    TT_FATAL(
        !(input_tensor_shape[-2] % tt::constants::TILE_HEIGHT),
        "Input tensor height ({}) must be divisible by tile height ({}).",
        input_tensor_shape[-2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        !(input_tensor_shape[-1] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[-1],
        tt::constants::TILE_WIDTH);

    const auto [normalized_dim, input_tensor_C, input_tensor_B] =
        (input_tensor_shape.rank() == 2) ? operations::experimental::ccl::detail::map_2d_to_4d(dim)
                                         : operations::experimental::ccl::detail::map_nd_to_4d(input_tensor_shape, dim);
    const uint32_t input_tensor_Ht = input_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;

    uint32_t slice_B = input_tensor_B;
    uint32_t slice_C = input_tensor_C;
    uint32_t slice_Ht = input_tensor_Ht;
    uint32_t slice_Wt = input_tensor_Wt;
    if (normalized_dim == 0) {
        slice_B /= ring_size;
    } else if (normalized_dim == 1) {
        slice_C /= ring_size;
    } else if (normalized_dim == 2) {
        slice_Ht /= ring_size;
    } else if (normalized_dim == 3) {
        slice_Wt /= ring_size;
    } else {
        TT_FATAL(
            false, "reduce_scatter_minimal_async ring implementation only supports scattering on dim 0, 1, 2, or 3");
    }

    TT_FATAL(
        !(fuse_op && normalized_dim == 0),
        "reduce_scatter_minimal_async ring implementation can't be fused with matmul when scattering on dim 0");

    const uint32_t input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;
    const uint32_t input_batch_num_pages = input_tensor_num_pages / input_tensor_B;
    const uint32_t output_batch_num_pages = output_tensor_num_pages / slice_B;
    const uint32_t input_channel_num_pages = input_batch_num_pages / input_tensor_C;
    const uint32_t output_channel_num_pages = output_batch_num_pages / slice_C;

    // scatter-write currently only supports 2 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 2;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t tile_granularity = num_tiles_to_write_per_packet < 4 ? 4 * num_tiles_to_write_per_packet : 8;
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_compute_output_config);

    bool input_is_sharded = input_tensor.is_sharded();
    bool intermediate_is_sharded = intermediate_tensor.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;

    if (input_is_sharded) {
        reader_compute_defines["INPUT_IS_SHARDED"] = "1";
    }
    if (intermediate_is_sharded) {
        reader_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
        writer_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }

    // KERNEL CREATION
    std::vector<size_t> mux_termination_signal_addresses;
    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }

    // Kernel Runtime Args
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    const auto num_full_size_channels = num_workers_per_direction;
    constexpr auto num_header_only_channels = 0;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        0,
        buffer_size_bytes_full_size_channel,
        mux_base_l1_address);

    // Fabric mux kernel
    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    std::vector<uint32_t> sender_reader_compile_args =
        operations::experimental::ccl::detail::get_ring_reader_compile_args(
            ring_index,
            ring_size,
            input_cb_index,
            intermediate_cb_index,
            reader_output_cb_index,
            tile_granularity,
            page_size,
            output_tensor_num_pages,
            input_batch_num_pages,
            input_channel_num_pages,
            input_tensor_B,
            input_tensor_Wt,
            slice_B,
            slice_C,
            slice_Ht,
            slice_Wt,
            fuse_op,
            normalized_dim);

    if (input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_reader_compile_args);
    }

    std::string sender_reader_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_ring_reduce_scatter_minimal_async_reader.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/ring_reduce_scatter_minimal_async_reader.cpp";

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_reader_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));

    // Writer
    std::vector<uint32_t> sender_writer_compile_args =
        operations::experimental::ccl::detail::get_ring_writer_compile_args(
            ring_index,
            ring_size,
            compute_output_cb_index,
            reader_output_cb_index,
            tile_granularity,
            page_size,
            num_tiles_to_write_per_packet,
            output_tensor_num_pages,
            output_batch_num_pages,
            input_batch_num_pages,
            input_channel_num_pages,
            output_channel_num_pages,
            input_tensor_B,
            input_tensor_Wt,
            slice_B,
            slice_C,
            slice_Ht,
            slice_Wt,
            normalized_dim);

    append_fabric_mux_connection_ct_args(
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        num_workers_per_direction,
        sender_writer_compile_args);

    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());

    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_writer_compile_args);
    }
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
    }

    std::string sender_writer_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_ring_reduce_scatter_minimal_async_writer.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/ring_reduce_scatter_minimal_async_writer.cpp";

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_writer_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));

    // Reduce kernel
    auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
    sender_reduce_kernel_config.compile_args = operations::experimental::ccl::detail::get_ring_reduce_compile_args(
        input_cb_index,
        intermediate_cb_index,
        compute_output_cb_index,
        tile_granularity,
        ring_size,
        input_tensor_B,
        slice_B,
        slice_C,
        normalized_dim);

    std::string sender_reduce_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_ring_reduction.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/ring_reduction.cpp";

    auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
        program, sender_reduce_kernel_path, sender_worker_core_range_set, sender_reduce_kernel_config);

    auto worker_core_iter = sender_worker_core_range_set.ranges().cbegin();
    auto mux_core_iter = mux_core_range_set.ranges().cbegin();
    auto termination_master_core_iter = termination_master_core_ranges.cbegin();
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            CoreCoord mux_virtual_core = {0, 0};
            if (mux_connection_valid(dir)) {
                auto mux_logical_core = *((mux_core_iter++)->begin());
                mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

                std::vector<uint32_t> mux_rt_args = {};
                const auto src_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                if (dir) {  // forward
                    const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                } else {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                }
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            }

            auto termination_master_logical_core = *((termination_master_core_iter++)->begin());
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                auto core = *((worker_core_iter++)->begin());
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

                uint32_t worker_id = (link * num_workers_per_direction) + worker;
                uint32_t num_workers = num_links * num_workers_per_direction;

                auto [start_tiles_read, start_tiles_to_read, start_pages_read_in_row, start_row_offset] =
                    operations::experimental::ccl::detail::get_tile_offsets(
                        worker_id,
                        num_workers,
                        output_batch_num_pages,
                        output_channel_num_pages,
                        slice_Wt,
                        input_tensor_Wt,
                        normalized_dim);

                // for dim 0 scatters we process each slice in batches
                // for all other dims we process each slice in channels
                uint32_t tiles_to_process_per_slice =
                    (start_tiles_to_read - start_tiles_read) * (normalized_dim == 0 ? slice_B : slice_C);
                uint32_t chunks_per_sync_val =
                    chunks_per_sync.value_or(operations::experimental::ccl::detail::default_chunks_per_sync(
                        topology, tiles_to_process_per_slice, tile_granularity));
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),         // input_tensor_address
                    intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                    semaphore.at(dir).address(),              // out_ready_semaphore
                    dir,                                      // direction
                    chunks_per_sync_val,                      // chunks_per_sync
                    start_tiles_read,                         // start_tiles_read
                    start_tiles_to_read,                      // start_tiles_to_read
                    start_pages_read_in_row,                  // start_pages_read_in_row
                    start_row_offset,                         // start_row_offset (unused by dim0 kernel)
                };
                if (input_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, reader_rt_args);
                }
                if (fuse_op) {
                    fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }

                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                // Writer RT args
                std::vector<uint32_t> writer_rt_args = {
                    intermediate_tensor.buffer()->address(),                     // intermediate_tensor_address
                    output_tensor.buffer()->address(),                           // output_tensor_address
                    virtual_core.x,                                              // out_ready_sem_noc0_x
                    virtual_core.y,                                              // out_ready_sem_noc0_y
                    semaphore.at(dir).address(),                                 // out_ready_fwd_semaphore
                    semaphore.at(num_directions_per_link).address(),             // batch_ready_semaphore
                    barrier_semaphore.has_value() && !using_persistent_buffers,  // use_barrier_sem
                    barrier_semaphore.has_value()                                // barrier_sem
                        ? barrier_semaphore.value().address()
                        : 0,
                    dir,                      // direction
                    chunks_per_sync_val,      // chunks_per_sync
                    start_pages_read_in_row,  // start_pages_read_in_row (unused by dim0 kernel)
                    start_row_offset,         // start_row_offset (unused by dim0 kernel)
                    start_tiles_read,         // start_tiles_read
                    start_tiles_to_read,      // tiles_to_read

                };
                append_fabric_mux_connection_rt_args(
                    mux_connection_valid(dir),
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_kernel_config,
                    core,
                    worker,
                    worker == 0,
                    termination_master_virtual_core,
                    program,
                    writer_rt_args);
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, writer_rt_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
                }
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);

                std::vector<uint32_t> reduce_rt_args = {
                    start_tiles_read,     // start_tiles_read
                    start_tiles_to_read,  // start_tiles_to_read
                    dir};                 // dir
                tt::tt_metal::SetRuntimeArgs(program, sender_reduce_kernel_id, {core}, reduce_rt_args);
            }
        }
    }

    return {
        reader_kernel_id,
        writer_kernel_id,
        all_cores,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link};
}

void ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output) {
    // update senders
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                uint32_t mux_core_offset = (link * num_cores_per_link) +
                                           (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                std::vector<std::vector<RuntimeArgsData>> reader_runtime_args =
                    GetRuntimeArgs(program, reader_kernel_id);
                std::vector<std::vector<RuntimeArgsData>> writer_runtime_args =
                    GetRuntimeArgs(program, writer_kernel_id);

                // sender reader
                auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
                worker_reader_sender_runtime_args[2] = semaphore.at(dir).address();
                // sender writer
                auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                worker_writer_sender_runtime_args[1] = output.buffer()->address();
                worker_writer_sender_runtime_args[4] = semaphore.at(dir).address();
                worker_writer_sender_runtime_args[5] = semaphore.at(num_directions_per_link).address();

                if (barrier_semaphore.has_value()) {
                    worker_writer_sender_runtime_args[7] = barrier_semaphore.value().address();
                }
            }
        }
    }
}

ReduceScatterProgramArtifacts build_line_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset) {
    /**
     * Line Reduce Scatter
     *
     *   IN 0     IN 1     IN 2     IN 3            OUT 0    OUT 1    OUT 2    OUT 3
     *   C0       C1       C2       C3              C0       C1       C2       C3
     *  ┌────┐   ┌────┐   ┌────┐   ┌────┐          ┌────┐   ......   ......   ......
     *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
     *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
     *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
     *  ├────┤   ├────┤   ├────┤   ├────┤          └────┘   ┌────┐   ......   ......
     *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
     *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
     *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
     *  ├────┤   ├────┤   ├────┤   ├────┤  ────►   ......   └────┘   ┌────┐   ......
     *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
     *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
     *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
     *  ├────┤   ├────┤   ├────┤   ├────┤          ......   ......   └────┘   ┌────┐
     *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
     *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
     *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
     *  └────┘   └────┘   └────┘   └────┘          ......   ......   ......   └────┘
     *
     *
     * There are (ring_size - 1) algorithmic steps in Line Reduce Scatter.
     * Each device must send (num_forward_targets) partials forward and
     * (num_backward_targets) partials backward.
     *
     * On each step, a device will:
     * - if first device in a direction, send a slice in that direction
     * - otherwise, receive a slice, locally reduce it, and send the result in that direction
     *
     */
    auto* mesh_device = input_tensor.device();
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;

    // op hyperparams
    // Get worker cores
    // 2 senders (reader + core + writer) per direction (forward, backward) per link
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    uint32_t input_data_size_bytes = input_tensor.buffer()->size();
    uint32_t num_workers_per_direction =
        num_workers_per_direction_opt.value_or(operations::experimental::ccl::detail::default_workers(
            *mesh_device,
            sub_device_id,
            topology,
            input_data_size_bytes,
            num_links,
            ring_size,
            num_directions_per_link,
            num_mux_cores_per_direction_per_link));
    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    bool fuse_op = fused_op_signaler.has_value();

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        topology, sender_device_coord, forward_coord, backward_coord, mesh_device);
    auto [num_targets_forward, num_targets_backward] =
        ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, true);
    auto [mcast_forward_args, mcast_backward_args] = ccl::get_forward_backward_line_mcast_configuration(
        topology,
        sender_device_coord,
        forward_coord,
        backward_coord,
        num_targets_forward,
        num_targets_backward,
        mesh_device);

    uint32_t num_cores_per_link =
        operations::experimental::ccl::detail::reduce_scatter_minimal_async_core_count_per_link(
            num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    const auto [all_core_range, all_cores] =
        choose_worker_cores(num_links, num_cores_per_link, mesh_device, sub_device_id, core_grid_offset);

    const auto mux_connection_valid = [&backward_coord, &forward_coord](const uint32_t dir) {
        return (!dir && backward_coord.has_value()) || (dir && forward_coord.has_value());
    };

    std::vector<CoreRange> sender_worker_core_ranges;
    std::vector<CoreRange> mux_core_ranges;
    std::vector<CoreRange> termination_master_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (mux_connection_valid(dir)) {
                mux_core_ranges.emplace_back(mux_core);
            }
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                sender_worker_core_ranges.emplace_back(worker_core);
                if (worker == 0) {
                    termination_master_core_ranges.emplace_back(worker_core);
                }
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_core_ranges);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t max_scatter_write_pages = 2;
    const uint32_t max_dst_size = 8;  // TODO: generalize based on arch and fp32 acc
    uint32_t tiles_to_write_per_packet = std::min(num_pages_per_packet, max_scatter_write_pages);
    uint32_t tile_granularity = std::min(4 * num_pages_per_packet, max_dst_size);
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_compute_output_config);

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    TT_FATAL(
        !(input_tensor_shape[-2] % tt::constants::TILE_HEIGHT),
        "Input tensor height ({}) must be divisible by tile height ({}).",
        input_tensor_shape[-2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        !(input_tensor_shape[-1] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[-1],
        tt::constants::TILE_WIDTH);

    const auto [normalized_dim, input_tensor_C, input_tensor_B] =
        (input_tensor_shape.rank() == 2) ? operations::experimental::ccl::detail::map_2d_to_4d(dim)
                                         : operations::experimental::ccl::detail::map_nd_to_4d(input_tensor_shape, dim);
    const uint32_t input_tensor_Ht = input_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;

    uint32_t slice_B = input_tensor_B;
    uint32_t slice_C = input_tensor_C;
    uint32_t slice_Ht = input_tensor_Ht;
    uint32_t slice_Wt = input_tensor_Wt;
    if (normalized_dim == 0) {
        slice_B /= ring_size;
    } else if (normalized_dim == 1) {
        slice_C /= ring_size;
    } else if (normalized_dim == 2) {
        slice_Ht /= ring_size;
    } else if (normalized_dim == 3) {
        slice_Wt /= ring_size;
    } else {
        TT_FATAL(
            false, "reduce_scatter_minimal_async line implementation only supports scattering on dim 0, 1, 2, or 3");
    }

    TT_FATAL(
        !(fuse_op && normalized_dim == 0),
        "reduce_scatter_minimal_async line implementation can't be fused with matmul when scattering on dim 0");

    const uint32_t input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;
    const uint32_t input_batch_num_pages = input_tensor_num_pages / input_tensor_B;
    const uint32_t output_batch_num_pages = output_tensor_num_pages / slice_B;
    const uint32_t input_channel_num_pages = input_batch_num_pages / input_tensor_C;
    const uint32_t output_channel_num_pages = output_batch_num_pages / slice_C;

    bool input_is_sharded = input_tensor.is_sharded();
    bool intermediate_is_sharded = intermediate_tensor.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;

    if (input_is_sharded) {
        reader_compute_defines["INPUT_IS_SHARDED"] = "1";
    }
    if (intermediate_is_sharded) {
        reader_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
        writer_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        reader_compute_defines["OUTPUT_IS_SHARDED"] = "1";
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }

    // KERNEL CREATION
    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }

    // Common Kernel Runtime Args
    const bool sync_with_other_direction = !(is_first_chip || is_last_chip);
    uint32_t fwd_bwd_semaphore_address = tt::tt_metal::CreateSemaphore(program, sender_worker_core_range_set, 0);
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;

    const auto num_full_size_channels = num_workers_per_direction;
    const auto num_header_only_channels = 0;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    // Fabric mux kernel
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        0,
        buffer_size_bytes_full_size_channel,
        mux_base_l1_address);

    // mux kernel
    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    // Reader
    std::vector<uint32_t> sender_reader_compile_args =
        operations::experimental::ccl::detail::get_line_reader_compile_args(
            ring_index,
            ring_size,
            input_cb_index,
            intermediate_cb_index,
            reader_output_cb_index,
            tile_granularity,
            page_size,
            input_tensor_num_pages,
            output_tensor_num_pages,
            input_batch_num_pages,
            input_channel_num_pages,
            output_batch_num_pages,
            output_channel_num_pages,
            input_tensor_B,
            input_tensor_Wt,
            slice_B,
            slice_C,
            slice_Ht,
            slice_Wt,
            fuse_op,
            sync_with_other_direction,
            normalized_dim);

    if (input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_reader_compile_args);
    }

    std::string sender_reader_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_line_reduce_scatter_minimal_async_reader.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/line_reduce_scatter_minimal_async_reader.cpp";

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_reader_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));

    // Writer
    std::vector<uint32_t> sender_writer_compile_args =
        operations::experimental::ccl::detail::get_line_writer_compile_args(
            ring_size,
            compute_output_cb_index,
            reader_output_cb_index,
            tile_granularity,
            page_size,
            tiles_to_write_per_packet,
            input_tensor_num_pages,
            output_tensor_num_pages,
            input_batch_num_pages,
            input_channel_num_pages,
            output_batch_num_pages,
            output_channel_num_pages,
            input_tensor_B,
            input_tensor_Wt,
            slice_B,
            slice_C,
            slice_Ht,
            slice_Wt,
            normalized_dim,
            sync_with_other_direction);

    append_fabric_mux_connection_ct_args(
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        num_workers_per_direction,
        sender_writer_compile_args);

    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());

    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_writer_compile_args);
    }
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
    }

    std::string sender_writer_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_line_reduce_scatter_minimal_async_writer.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/line_reduce_scatter_minimal_async_writer.cpp";

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_writer_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));

    // Reduce kernel
    auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
    sender_reduce_kernel_config.compile_args = operations::experimental::ccl::detail::get_line_reduce_compile_args(
        input_cb_index,
        intermediate_cb_index,
        compute_output_cb_index,
        tile_granularity,
        input_tensor_B,
        slice_B,
        slice_C,
        normalized_dim);

    std::string sender_reduce_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_line_reduction.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/line_reduction.cpp";

    auto reduce_kernel_id = tt::tt_metal::CreateKernel(
        program, sender_reduce_kernel_path, sender_worker_core_range_set, sender_reduce_kernel_config);

    auto worker_core_iter = sender_worker_core_range_set.ranges().cbegin();
    auto mux_core_iter = mux_core_range_set.ranges().cbegin();
    auto termination_master_core_iter = termination_master_core_ranges.cbegin();
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const bool is_forward = dir;
            CoreCoord mux_virtual_core = {0, 0};
            if (mux_connection_valid(dir)) {
                auto mux_logical_core = *((mux_core_iter++)->begin());
                mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);
                std::vector<uint32_t> mux_rt_args = {};
                const auto src_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                if (dir) {  // forward
                    const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                } else {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                }
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            }

            auto termination_master_logical_core = *((termination_master_core_iter++)->begin());
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                auto core = *((worker_core_iter++)->begin());
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

                // FWD core needs BWD core coordinate for fwd/bwd final reduction sync.
                // For final synchronization, each core needs to know the coordinate of the opposite direction's core.
                uint32_t opposite_mux_core_offset =
                    (link * num_cores_per_link) +
                    ((1 - dir) * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
                uint32_t opposite_core_idx = opposite_mux_core_offset + num_mux_cores_per_direction_per_link + worker;
                auto opposite_core = all_cores[opposite_core_idx];
                auto opposite_core_coord = mesh_device->worker_core_from_logical_core(opposite_core);

                /**
                 * Every chip has a final reduction step. On the ends, there is only one input to reduce.
                 * In the middle, you must reduce from both input directions.
                 * FWD/BWD readers need to synchronize in order to avoid race conditions.
                 *
                 * We'll say that FWD always leads, then signals BWD to follow.
                 */
                const bool is_first_device_in_direction = is_forward ? is_first_chip : is_last_chip;
                const int num_targets_in_direction = is_forward ? num_targets_forward : num_targets_backward;
                // The number of reduction steps is 0 for chips on the ends, or num_targets_in_direction for chips in
                // the middle
                const int num_intermediate_reduction_steps =
                    is_first_device_in_direction ? 0 : num_targets_in_direction;
                const bool do_final_reduction = !is_first_device_in_direction;
                const int num_total_reduction_steps = num_intermediate_reduction_steps + (do_final_reduction ? 1 : 0);

                const uint32_t worker_id = (link * num_workers_per_direction) + worker;
                const uint32_t num_workers = num_links * num_workers_per_direction;
                const auto [start_tiles_read, start_tiles_to_read, start_pages_read_in_row, start_row_offset] =
                    operations::experimental::ccl::detail::get_tile_offsets(
                        worker_id,
                        num_workers,
                        output_batch_num_pages,
                        output_channel_num_pages,
                        slice_Wt,
                        input_tensor_Wt,
                        normalized_dim);

                // for dim 0 scatters we process each slice in batches
                // for all other dims we process each slice in channels
                uint32_t tiles_to_process_per_slice =
                    (start_tiles_to_read - start_tiles_read) * (normalized_dim == 0 ? slice_B : slice_C);
                uint32_t chunks_per_sync_val =
                    chunks_per_sync.value_or(operations::experimental::ccl::detail::default_chunks_per_sync(
                        topology, tiles_to_process_per_slice, tile_granularity));
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                // Reader RT args
                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),         // input_tensor_address
                    intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                    output_tensor.buffer()->address(),        // output_tensor_address
                    semaphore.at(0).address(),                // remote transfer sync semaphore
                    fwd_bwd_semaphore_address,
                    is_forward,                    // is_forward
                    is_first_device_in_direction,  // is_first_device_in_direction
                    num_targets_in_direction,      // num_targets_in_direction
                    do_final_reduction,            // do_final_reduction
                    chunks_per_sync_val,           // chunks_per_sync
                    start_tiles_read,              // start_tiles_read
                    start_tiles_to_read,           // start_tiles_to_read
                    start_pages_read_in_row,       // start_pages_read_in_row (unused by dim0 kernel)
                    start_row_offset,              // start_row_offset (unused by dim0 kernel)
                };

                if (input_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, reader_rt_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, reader_rt_args);
                }
                if (fuse_op) {
                    fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                std::vector<uint32_t> writer_rt_args = {
                    intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                    output_tensor.buffer()->address(),        // output_tensor_address
                    virtual_core.x,                           // out_ready_sem_noc0_x
                    virtual_core.y,                           // out_ready_sem_noc0_y
                    semaphore.at(0).address(),                // remote transfer sync semaphore
                    fwd_bwd_semaphore_address,
                    opposite_core_coord.x,
                    opposite_core_coord.y,
                    barrier_semaphore.has_value() && !using_persistent_buffers,  // use_barrier_sem
                    barrier_semaphore.has_value()                                // synchronize barrier semaphore
                        ? barrier_semaphore.value().address()
                        : 0,
                    is_forward,                    // is_forward
                    is_first_device_in_direction,  // is_first_device_in_direction
                    num_targets_in_direction,      // num_targets_in_direction
                    do_final_reduction,            // do_final_reduction
                    chunks_per_sync_val,           // chunks_per_sync
                    start_pages_read_in_row,       // start_pages_read_in_row
                    start_row_offset,              // start_row_offset
                    start_tiles_read,              // start_tiles_read
                    start_tiles_to_read,           // start_tiles_to_read
                };
                append_fabric_mux_connection_rt_args(
                    mux_connection_valid(dir),
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_kernel_config,
                    core,
                    worker,
                    worker == 0,
                    termination_master_virtual_core,
                    program,
                    writer_rt_args);
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, writer_rt_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
                }

                tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);

                std::vector<uint32_t> reduce_rt_args = {
                    num_total_reduction_steps,
                    start_tiles_read,
                    start_tiles_to_read,
                };
                tt::tt_metal::SetRuntimeArgs(program, reduce_kernel_id, {core}, reduce_rt_args);
            }
        }
    }

    return {
        reader_kernel_id,
        writer_kernel_id,
        all_cores,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link};
}

void line_reduce_scatter_minimal_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output) {
    // update senders
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                uint32_t mux_core_offset = (link * num_cores_per_link) +
                                           (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                std::vector<std::vector<RuntimeArgsData>> reader_runtime_args =
                    GetRuntimeArgs(program, reader_kernel_id);
                std::vector<std::vector<RuntimeArgsData>> writer_runtime_args =
                    GetRuntimeArgs(program, writer_kernel_id);

                // sender reader
                auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
                worker_reader_sender_runtime_args[2] = output.buffer()->address();
                worker_reader_sender_runtime_args[3] = semaphore.at(0).address();
                // sender writer
                auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                worker_writer_sender_runtime_args[1] = output.buffer()->address();
                worker_writer_sender_runtime_args[4] = semaphore.at(0).address();

                if (barrier_semaphore.has_value()) {
                    worker_writer_sender_runtime_args[9] = barrier_semaphore.value().address();
                }
            }
        }
    }
}

}  // namespace ttnn

// Implementations for the TMP namespace - wrappers to ttnn namespace functions
namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::detail {

ReduceScatterProgramArtifacts build_ring_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset) {
    return ::ttnn::build_ring_reduce_scatter_minimal_async_program_artifacts(
        program,
        input_tensor,
        intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset);
}

ReduceScatterProgramArtifacts build_line_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset) {
    return ::ttnn::build_line_reduce_scatter_minimal_async_program_artifacts(
        program,
        input_tensor,
        intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset);
}

void ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output) {
    ::ttnn::ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        all_cores,
        num_links,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link,
        barrier_semaphore,
        semaphore,
        input,
        intermed,
        output);
}

void line_reduce_scatter_minimal_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output) {
    ::ttnn::line_reduce_scatter_minimal_async_helper_override_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        all_cores,
        num_links,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link,
        barrier_semaphore,
        semaphore,
        input,
        intermed,
        output);
}

// Mesh Workload Factory implementations
RingReduceScatterMeshWorkloadFactory::cached_mesh_workload_t RingReduceScatterMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return {std::move(mesh_workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<RingReduceScatterMeshWorkloadFactory::shared_variables_t>
RingReduceScatterMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto& intermediate_tensor = tensor_return_value.at(0);
    auto& output_tensor = tensor_return_value.at(1);

    const auto forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const auto backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "forward_coord or backward_coord is null");

    const uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> fused_op_signaler = std::nullopt;
    tt::tt_metal::Program program{};
    auto shared_vars = build_ring_reduce_scatter_minimal_async_program_artifacts(
        program,
        input_tensor,
        intermediate_tensor,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        output_tensor,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        ring_index,
        operation_attributes.topology,
        operation_attributes.semaphore,
        operation_attributes.barrier_semaphore,
        operation_attributes.using_persistent_buffers,
        operation_attributes.sub_device_id,
        fused_op_signaler,
        operation_attributes.chunks_per_sync,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        CoreCoord(0, 0));

    return {std::move(program), std::move(shared_vars)};
}

void RingReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& intermediate = tensor_return_value.at(0);
    const auto& output = tensor_return_value.at(1);

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
            program,
            shared_vars.reader_kernel_id,
            shared_vars.writer_kernel_id,
            shared_vars.all_cores,
            operation_attributes.num_links,
            shared_vars.num_directions_per_link,
            shared_vars.num_workers_per_direction,
            shared_vars.num_mux_cores_per_direction_per_link,
            shared_vars.num_cores_per_link,
            operation_attributes.barrier_semaphore,
            operation_attributes.semaphore,
            input,
            intermediate,
            output);
    }
}

LineReduceScatterMeshWorkloadFactory::cached_mesh_workload_t LineReduceScatterMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return {std::move(mesh_workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<LineReduceScatterMeshWorkloadFactory::shared_variables_t>
LineReduceScatterMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto& intermediate_tensor = tensor_return_value.at(0);
    auto& output_tensor = tensor_return_value.at(1);

    const auto forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const auto backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "forward_coord or backward_coord is null");

    const uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> fused_op_signaler = std::nullopt;
    tt::tt_metal::Program program{};
    auto shared_vars = build_line_reduce_scatter_minimal_async_program_artifacts(
        program,
        input_tensor,
        intermediate_tensor,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        output_tensor,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        ring_index,
        operation_attributes.topology,
        operation_attributes.semaphore,
        operation_attributes.barrier_semaphore,
        operation_attributes.using_persistent_buffers,
        operation_attributes.sub_device_id,
        fused_op_signaler,
        operation_attributes.chunks_per_sync,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        CoreCoord(0, 0));

    return {std::move(program), std::move(shared_vars)};
}

void LineReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& intermediate = tensor_return_value.at(0);
    const auto& output = tensor_return_value.at(1);

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        TT_FATAL(
            operation_attributes.topology == ttnn::ccl::Topology::Linear,
            "LineReduceScatterMeshWorkloadFactory expects Linear topology");

        line_reduce_scatter_minimal_async_helper_override_runtime_arguments(
            program,
            shared_vars.reader_kernel_id,
            shared_vars.writer_kernel_id,
            shared_vars.all_cores,
            operation_attributes.num_links,
            shared_vars.num_directions_per_link,
            shared_vars.num_workers_per_direction,
            shared_vars.num_mux_cores_per_direction_per_link,
            shared_vars.num_cores_per_link,
            operation_attributes.barrier_semaphore,
            operation_attributes.semaphore,
            input,
            intermediate,
            output);
    }
}

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::detail
