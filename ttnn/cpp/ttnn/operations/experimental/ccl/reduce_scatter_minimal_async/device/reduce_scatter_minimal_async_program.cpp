// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
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
    ttnn::ccl::Topology topology, uint32_t tiles_to_read, uint32_t tiles_read, uint32_t tile_granularity) {
    // For Line, as early as 20 chunks per sync we get statistically significant performance improvements
    // For Ring there is no statistically significant performance improvements until 80 chunks per sync, which beats the
    // default value of syncing once This was determined by the sweep test:
    // tests/ttnn/multidevice_perf_tests/test_reduce_scatter_hyperparameter_sweep_perf_galaxy.py
    TT_FATAL(topology == ttnn::ccl::Topology::Ring || topology == ttnn::ccl::Topology::Linear, "Invalid topology");
    constexpr uint32_t RING_DEFAULT_CHUNKS_PER_SYNC = 80;
    constexpr uint32_t LINEAR_DEFAULT_CHUNKS_PER_SYNC = 20;
    uint32_t default_value =
        topology == ttnn::ccl::Topology::Ring ? RING_DEFAULT_CHUNKS_PER_SYNC : LINEAR_DEFAULT_CHUNKS_PER_SYNC;
    uint32_t total_chunks = std::max((tiles_to_read - tiles_read) / tile_granularity / 2, (uint32_t)1);
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
    const uint32_t dir,
    const uint32_t chunks_per_sync_val,
    const uint32_t normalized_dim,
    const uint32_t start_pages_read_in_row,
    const uint32_t start_row_offset,
    const uint32_t start_tiles_read,
    const uint32_t start_tiles_to_read) {
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
            dir,                      // direction
            chunks_per_sync_val,      // chunks_per_sync
            start_tiles_read,         // start_tiles_read
            start_tiles_to_read       // start_tiles_to_read
        };
    } else {
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
            fuse_op,                  // fused op
            dir,                      // direction
            chunks_per_sync_val,      // chunks_per_sync
            normalized_dim,           // dim normalized to 4D
            start_pages_read_in_row,  // start_pages_read_in_row
            start_row_offset,         // start_row_offset
            start_tiles_read,         // start_tiles_read
            start_tiles_to_read       // start_tiles_to_read
        };
    }
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
    const uint32_t dir,
    const uint32_t chunks_per_sync_val,
    const uint32_t normalized_dim,
    const uint32_t start_pages_read_in_row,
    const uint32_t start_row_offset,
    const uint32_t start_tiles_read,
    const uint32_t start_tiles_to_read) {
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
            dir,                            // direction
            chunks_per_sync_val,            // chunks_per_sync
            start_tiles_read,               // start_tiles_read
            start_tiles_to_read,            // tiles_to_read
        };
    } else {
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
            input_tensor_Wt,                // input_tensor_Wt
            slice_C,                        // slice_C
            slice_Ht,                       // slice_Ht
            slice_Wt,                       // slice_Wt
            dir,                            // direction
            chunks_per_sync_val,            // chunks_per_sync
            normalized_dim,                 // dim normalized to 4D
            start_pages_read_in_row,        // start_pages_read_in_row
            start_row_offset,               // start_row_offset
            start_tiles_read,               // start_tiles_read
            start_tiles_to_read,            // tiles_to_read
        };
    }
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
    const uint32_t dir,
    const uint32_t start_tiles_read,
    const uint32_t start_tiles_to_read,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            input_cb_index,           // input_cb_id
            intermediate_cb_index,    // intermediate_cb
            compute_output_cb_index,  // output_cb
            tile_granularity,         // tile_granularity
            ring_size,                // ring_size
            slice_B,                  // slice_B
            dir,                      // dir
            start_tiles_read,         // start_tiles_read
            start_tiles_to_read       // start_tiles_to_read
        };
    } else {
        return {
            input_cb_index,           // input_cb_id
            intermediate_cb_index,    // intermediate_cb
            compute_output_cb_index,  // output_cb
            tile_granularity,         // tile_granularity
            ring_size,                // ring_size
            input_tensor_B,           // input_tensor_B
            slice_C,                  // slice_C
            dir,                      // dir
            start_tiles_read,         // start_tiles_read
            start_tiles_to_read       // start_tiles_to_read
        };
    }
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
    const uint32_t is_forward,
    const uint32_t is_first_device_in_direction,
    const uint32_t num_targets_in_direction,
    const uint32_t do_final_reduction,
    const uint32_t sync_with_other_direction,
    const uint32_t chunks_per_sync_val,
    const uint32_t normalized_dim,
    const uint32_t start_pages_read_in_row,
    const uint32_t start_row_offset,
    const uint32_t start_tiles_read,
    const uint32_t start_tiles_to_read) {
    if (normalized_dim == 0) {
        return {
            ring_index,                    // my_chip_id
            ring_size,                     // ring_size
            input_cb_index,                // cb_input_id
            intermediate_cb_index,         // cb_intermediate_id
            reader_output_cb_index,        // cb_reader_output_id
            tile_granularity,              // tile_granularity
            page_size,                     // page_size
            input_tensor_num_pages,        // input_num_pages
            output_tensor_num_pages,       // output_num_pages
            input_batch_num_pages,         // batch_num_pages
            slice_B,                       // slice_B
            is_forward,                    // is_forward
            is_first_device_in_direction,  // is_first_device_in_direction
            num_targets_in_direction,      // num_targets_in_direction
            do_final_reduction,            // do_final_reduction
            sync_with_other_direction,     // sync_with_other_direction
            chunks_per_sync_val,           // chunks_per_sync
            start_tiles_read,              // start_tiles_read
            start_tiles_to_read            // start_tiles_to_read
        };
    } else {
        return {
            ring_index,                    // my_chip_id
            ring_size,                     // ring_size
            input_cb_index,                // cb_input_id
            intermediate_cb_index,         // cb_intermediate_id
            reader_output_cb_index,        // cb_reader_output_id
            tile_granularity,              // tile_granularity
            page_size,                     // page_size
            input_tensor_num_pages,        // input_num_pages
            input_batch_num_pages,         // input_batch_num_pages
            input_channel_num_pages,       // input_channel_num_pages
            output_batch_num_pages,        // output_batch_num_pages
            output_channel_num_pages,      // output_channel_num_pages
            input_tensor_B,                // input_tensor_B
            input_tensor_Wt,               // input_tensor_Wt
            slice_C,                       // slice_C
            slice_Ht,                      // slice_Ht
            slice_Wt,                      // slice_Wt
            fuse_op,                       // fuse_op
            is_forward,                    // is_forward
            is_first_device_in_direction,  // is_first_device_in_direction
            num_targets_in_direction,      // num_targets_in_direction
            do_final_reduction,            // do_final_reduction
            sync_with_other_direction,     // sync_with_other_direction
            chunks_per_sync_val,           // chunks_per_sync
            normalized_dim,                // dim
            start_pages_read_in_row,       // start_pages_read_in_row
            start_row_offset,              // start_row_offset
            start_tiles_read,              // start_tiles_read
            start_tiles_to_read            // start_tiles_to_read
        };
    }
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
    const uint32_t is_forward,
    const uint32_t is_first_device_in_direction,
    const uint32_t num_targets_in_direction,
    const uint32_t do_final_reduction,
    const uint32_t sync_with_other_direction,
    const uint32_t chunks_per_sync_val,
    const uint32_t normalized_dim,
    const uint32_t start_pages_read_in_row,
    const uint32_t start_row_offset,
    const uint32_t start_tiles_read,
    const uint32_t start_tiles_to_read) {
    if (normalized_dim == 0) {
        return {
            ring_size,                     // ring_size
            compute_output_cb_index,       // cb_compute_output_id
            reader_output_cb_index,        // cb_reader_output_id
            tile_granularity,              // tile_granularity
            page_size,                     // page_size
            tiles_to_write_per_packet,     // contig_pages_advanced
            input_tensor_num_pages,        // input_num_pages
            output_tensor_num_pages,       // output_num_pages
            input_batch_num_pages,         // batch_num_pages
            slice_B,                       // slice_B
            is_forward,                    // is_forward
            is_first_device_in_direction,  // is_first_device_in_direction
            num_targets_in_direction,      // num_targets_in_direction
            do_final_reduction,            // do_final_reduction
            sync_with_other_direction,     // sync_with_other_direction
            chunks_per_sync_val,           // chunks_per_sync
            start_tiles_read,              // start_tiles_read
            start_tiles_to_read,           // start_tiles_to_read
        };
    } else {
        return {
            ring_size,                     // ring_size
            compute_output_cb_index,       // cb_compute_output_id
            reader_output_cb_index,        // cb_reader_output_id
            tile_granularity,              // tile_granularity
            page_size,                     // page_size
            tiles_to_write_per_packet,     // contig_pages_advanced
            input_tensor_num_pages,        // input_num_pages
            input_batch_num_pages,         // input_batch_num_pages
            input_channel_num_pages,       // input_channel_num_pages
            output_batch_num_pages,        // output_batch_num_pages
            output_channel_num_pages,      // output_channel_num_pages
            input_tensor_B,                // input_tensor_b
            input_tensor_Wt,               // input_tensor_Wt
            slice_C,                       // slice_C
            slice_Ht,                      // slice_Ht
            slice_Wt,                      // slice_Wt
            is_forward,                    // is_forward
            is_first_device_in_direction,  // is_first_device_in_direction
            num_targets_in_direction,      // num_targets_in_direction
            do_final_reduction,            // do_final_reduction
            sync_with_other_direction,     // sync_with_other_direction
            chunks_per_sync_val,           // chunks_per_sync
            normalized_dim,                // dim
            start_pages_read_in_row,       // start_pages_read_in_row
            start_row_offset,              // start_row_offset
            start_tiles_read,              // start_tiles_read
            start_tiles_to_read,           // start_tiles_to_read
        };
    }
}

std::vector<uint32_t> get_line_reduce_compile_args(
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t compute_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t input_tensor_B,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t num_total_reduction_steps,
    const uint32_t start_tiles_read,
    const uint32_t start_tiles_to_read,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            input_cb_index,
            intermediate_cb_index,
            compute_output_cb_index,
            tile_granularity,
            slice_B,
            num_total_reduction_steps,
            start_tiles_read,
            start_tiles_to_read};
    } else {
        return {
            input_cb_index,
            intermediate_cb_index,
            compute_output_cb_index,
            tile_granularity,
            input_tensor_B,
            slice_C,
            num_total_reduction_steps,
            start_tiles_read,
            start_tiles_to_read};
    }
}

}  // namespace operations::experimental::ccl::detail

using namespace ccl;

void append_fabric_mux_connection_ct_args(
    const bool is_termination_master,
    const CoreCoord& mux_virtual_core,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    uint32_t worker_id,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& writer_ct_args) {
    writer_ct_args.push_back(is_termination_master);
    writer_ct_args.push_back(mux_virtual_core.x);
    writer_ct_args.push_back(mux_virtual_core.y);
    writer_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));
    writer_ct_args.push_back(mux_kernel_config.get_buffer_size_bytes(channel_type));
    writer_ct_args.push_back(mux_kernel_config.get_channel_base_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_connection_info_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_connection_handshake_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_flow_control_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_buffer_index_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_status_address());
    writer_ct_args.push_back(mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_termination_signal_address());
}

void append_fabric_mux_connection_rt_args(
    const bool& mux_connection_valid,
    const CoreCoord& worker_logical_core,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    uint32_t num_workers_per_direction,
    std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(mux_connection_valid);
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(termination_master_virtual_core.x);
    worker_rt_args.push_back(termination_master_virtual_core.y);
    worker_rt_args.push_back(num_workers_per_direction);
}

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async(
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    const std::optional<uint32_t> chunks_per_sync,
    const std::optional<uint32_t> num_workers_per_link,
    const std::optional<uint32_t> num_buffers_per_channel) {
    tt::tt_metal::Program program{};
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler> empty_fused_op_signaler;

    return reduce_scatter_minimal_async_helper(
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
        empty_fused_op_signaler,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset) {
    if (topology == ccl::Topology::Ring) {
        return ring_reduce_scatter_minimal_async_helper(
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
            num_workers_per_link,
            num_buffers_per_channel);
    } else {
        TT_FATAL(topology == ccl::Topology::Linear, "Must be line or ring");
        return line_reduce_scatter_minimal_async_helper(
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
            num_workers_per_link,
            num_buffers_per_channel,
            core_grid_offset);
    }
}

ReduceScatterProgramArtifacts build_ring_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    auto mesh_device = input_tensor.device();
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
    std::set<CoreRange> sender_worker_core_ranges;
    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;
    std::set<CoreRange> mux_forward_core_ranges;
    std::set<CoreRange> mux_backward_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (dir) {
                mux_forward_core_ranges.insert(CoreRange(mux_core));
            } else {
                mux_backward_core_ranges.insert(CoreRange(mux_core));
            }
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                if (dir) {
                    sender_forward_core_ranges.insert(CoreRange(worker_core));
                } else {
                    sender_backward_core_ranges.insert(CoreRange(worker_core));
                }
                sender_worker_core_ranges.insert(CoreRange(worker_core));
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet sender_forward_core_range_set = CoreRangeSet(sender_forward_core_ranges);
    CoreRangeSet sender_backward_core_range_set = CoreRangeSet(sender_backward_core_ranges);
    CoreRangeSet mux_forward_core_range_set = CoreRangeSet(mux_forward_core_ranges);
    CoreRangeSet mux_backward_core_range_set = CoreRangeSet(mux_backward_core_ranges);

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
    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<KernelHandle> reduce_kernel_ids;
    std::vector<KernelHandle> mux_kernel_ids;
    std::vector<size_t> mux_termination_signal_addresses;
    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }

    // Kernel Runtime Args
    CoreCoord opposite_core_coord;
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            // Fabrix mux kernel
            uint32_t mux_core_offset = (link * num_cores_per_link) +
                                       (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
            CoreCoord mux_logical_core = all_cores[mux_core_offset];
            CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

            auto num_full_size_channels = num_workers_per_direction;
            auto num_header_only_channels = 0;
            size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
            auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
                num_full_size_channels,
                num_header_only_channels,
                num_buffers_full_size_channels,
                0,
                buffer_size_bytes_full_size_channel,
                mux_base_l1_address);

            auto mux_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                {mux_logical_core},
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
            mux_kernel_ids.push_back(mux_kernel_id);

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

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
                CoreCoord supplemental_core = all_cores
                    [(link * num_cores_per_link) +
                     ((1 - dir) * (num_mux_cores_per_direction_per_link + num_workers_per_direction)) +
                     num_mux_cores_per_direction_per_link + worker];
                opposite_core_coord = mesh_device->worker_core_from_logical_core(supplemental_core);

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

                uint32_t chunks_per_sync_val;
                if (normalized_dim == 0) {
                    chunks_per_sync_val =
                        chunks_per_sync.value_or(operations::experimental::ccl::detail::default_chunks_per_sync(
                            topology, start_tiles_to_read * slice_B, start_tiles_read * slice_B, tile_granularity));
                } else {
                    chunks_per_sync_val =
                        chunks_per_sync.value_or(operations::experimental::ccl::detail::default_chunks_per_sync(
                            topology, start_tiles_to_read * slice_C, start_tiles_read * slice_C, tile_granularity));
                }
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                // Reader CT args
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
                        dir,
                        chunks_per_sync_val,
                        normalized_dim,
                        start_pages_read_in_row,
                        start_row_offset,
                        start_tiles_read,
                        start_tiles_to_read);
                if (input_is_sharded) {
                    shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
                } else {
                    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
                } else {
                    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer())
                        .append_to(sender_reader_compile_args);
                }

                std::string sender_reader_kernel_path =
                    normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                          "device/kernels/dim_zero_ring_reduce_scatter_minimal_async_reader.cpp"
                                        : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                          "device/kernels/ring_reduce_scatter_minimal_async_reader.cpp";
                auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    sender_reader_kernel_path,
                    {core},
                    tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));
                reader_kernel_ids.push_back(worker_sender_reader_kernel_id);

                // Reader RT args
                std::vector<uint32_t> sender_reader_runtime_args = {
                    input_tensor.buffer()->address(),         // input_tensor_address
                    intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                    semaphore.at(dir).address(),              // out_ready_semaphore
                };
                if (input_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(input_tensor, sender_reader_runtime_args);
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, sender_reader_runtime_args);
                }
                if (fuse_op) {
                    fused_op_signaler->push_reduce_scatter_fused_op_rt_args(sender_reader_runtime_args);
                }
                tt::tt_metal::SetRuntimeArgs(
                    program, worker_sender_reader_kernel_id, {core}, sender_reader_runtime_args);

                CoreCoord termination_master_logical_core =
                    all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + 0];
                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                // Writer CT args
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
                        dir,
                        chunks_per_sync_val,
                        normalized_dim,
                        start_pages_read_in_row,
                        start_row_offset,
                        start_tiles_read,
                        start_tiles_to_read);
                append_fabric_mux_connection_ct_args(
                    worker == 0,
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    worker,
                    mux_kernel_config,
                    sender_writer_compile_args);
                if (dir) {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
                } else {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_writer_compile_args);
                } else {
                    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer())
                        .append_to(sender_writer_compile_args);
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
                auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    sender_writer_kernel_path,
                    {core},
                    tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));
                writer_kernel_ids.push_back(worker_sender_writer_kernel_id);

                // Writer RT args
                std::vector<uint32_t> sender_writer_runtime_args = {
                    intermediate_tensor.buffer()->address(),                     // intermediate_tensor_address
                    output_tensor.buffer()->address(),                           // output_tensor_address
                    virtual_core.x,                                              // out_ready_sem_noc0_x
                    virtual_core.y,                                              // out_ready_sem_noc0_y
                    semaphore.at(dir).address(),                                 // out_ready_fwd_semaphore
                    semaphore.at(num_directions_per_link).address(),             // batch_ready_semaphore
                    barrier_semaphore.has_value() && !using_persistent_buffers,  // use_barrier_sem
                    barrier_semaphore.has_value()                                // barrier_sem
                        ? barrier_semaphore.value().address()
                        : 0};
                append_fabric_mux_connection_rt_args(
                    true,
                    core,
                    program,
                    termination_master_virtual_core,
                    num_workers_per_direction,
                    sender_writer_runtime_args);
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, sender_writer_runtime_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, sender_writer_runtime_args);
                }
                tt::tt_metal::SetRuntimeArgs(
                    program, worker_sender_writer_kernel_id, {core}, sender_writer_runtime_args);

                // Reduce CT args
                std::vector<uint32_t> sender_reduce_compile_args =
                    operations::experimental::ccl::detail::get_ring_reduce_compile_args(
                        input_cb_index,
                        intermediate_cb_index,
                        compute_output_cb_index,
                        tile_granularity,
                        ring_size,
                        input_tensor_B,
                        slice_B,
                        slice_C,
                        dir,
                        start_tiles_read,
                        start_tiles_to_read,
                        normalized_dim);
                std::string sender_reduce_kernel_path =
                    normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                          "device/kernels/dim_zero_ring_reduction.cpp"
                                        : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                          "device/kernels/ring_reduction.cpp";
                auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    sender_reduce_kernel_path,
                    {core},
                    tt::tt_metal::ComputeConfig{.compile_args = sender_reduce_compile_args});
                reduce_kernel_ids.push_back(sender_reduce_kernel_id);

                // Reduce RT args
                std::vector<uint32_t> sender_reduce_runtime_args = {};
                tt::tt_metal::SetRuntimeArgs(program, sender_reduce_kernel_id, {core}, sender_reduce_runtime_args);
            }
        }
    }

    return {
        reader_kernel_ids,
        writer_kernel_ids,
        all_cores,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link};
}

void ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const std::vector<tt::tt_metal::KernelHandle>& reader_kernel_ids,
    const std::vector<tt::tt_metal::KernelHandle>& writer_kernel_ids,
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
    uint32_t core_idx = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                uint32_t mux_core_offset = (link * num_cores_per_link) +
                                           (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                std::vector<std::vector<RuntimeArgsData>> reader_runtime_args =
                    GetRuntimeArgs(program, reader_kernel_ids[core_idx]);
                std::vector<std::vector<RuntimeArgsData>> writer_runtime_args =
                    GetRuntimeArgs(program, writer_kernel_ids[core_idx]);

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

                core_idx++;
            }
        }
    }
}

tt::tt_metal::operation::ProgramWithCallbacks ring_reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    auto
        [reader_kernel_ids,
         writer_kernel_ids,
         all_cores,
         num_directions_per_link,
         num_workers_per_direction,
         num_mux_cores_per_direction_per_link,
         num_cores_per_link] =
            build_ring_reduce_scatter_minimal_async_program_artifacts(
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

    auto override_runtime_arguments_callback =
        [reader_kernel_ids,
         writer_kernel_ids,
         all_cores,
         num_links,
         num_directions_per_link,
         num_workers_per_direction,
         num_mux_cores_per_direction_per_link,
         num_cores_per_link](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[1];
            const auto& intermed = output_tensors[0];
            auto barrier_semaphore = static_cast<const ttnn::ReduceScatterMinimalAsync*>(operation)->barrier_semaphore;
            auto semaphore = static_cast<const ttnn::ReduceScatterMinimalAsync*>(operation)->semaphore;
            ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
                program,
                reader_kernel_ids,
                writer_kernel_ids,
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
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

ReduceScatterProgramArtifacts build_line_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    auto mesh_device = input_tensor.device();
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
    std::set<CoreRange> sender_worker_core_ranges;
    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;
    std::set<CoreRange> mux_forward_core_ranges;
    std::set<CoreRange> mux_backward_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (dir) {
                mux_forward_core_ranges.insert(CoreRange(mux_core));
            } else {
                mux_backward_core_ranges.insert(CoreRange(mux_core));
            }
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                if (dir) {
                    sender_forward_core_ranges.insert(CoreRange(worker_core));
                } else {
                    sender_backward_core_ranges.insert(CoreRange(worker_core));
                }
                sender_worker_core_ranges.insert(CoreRange(worker_core));
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet sender_forward_core_range_set = CoreRangeSet(sender_forward_core_ranges);
    CoreRangeSet sender_backward_core_range_set = CoreRangeSet(sender_backward_core_ranges);
    CoreRangeSet mux_forward_core_range_set = CoreRangeSet(mux_forward_core_ranges);
    CoreRangeSet mux_backward_core_range_set = CoreRangeSet(mux_backward_core_ranges);

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
    // Reader
    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<KernelHandle> reduce_kernel_ids;
    std::vector<KernelHandle> mux_kernel_ids;
    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }

    // Kernel Runtime Args
    uint32_t fwd_bwd_semaphore_address = tt::tt_metal::CreateSemaphore(program, sender_worker_core_range_set, 0);
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const bool is_forward = dir;

            // Fabrix mux kernel
            uint32_t mux_core_offset = (link * num_cores_per_link) +
                                       (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
            CoreCoord mux_logical_core = all_cores[mux_core_offset];
            CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

            auto num_full_size_channels = num_workers_per_direction;
            auto num_header_only_channels = 0;
            size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
            auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
                num_full_size_channels,
                num_header_only_channels,
                num_buffers_full_size_channels,
                0,
                buffer_size_bytes_full_size_channel,
                mux_base_l1_address);

            const bool mux_connection_valid =
                (dir && forward_coord.has_value()) || (!dir && backward_coord.has_value());
            if (mux_connection_valid) {
                auto mux_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                    {mux_logical_core},
                    tt::tt_metal::DataMovementConfig{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt::tt_metal::NOC::RISCV_0_default,
                        .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
                        .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
                mux_kernel_ids.push_back(mux_kernel_id);
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

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
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
                const bool sync_with_other_direction = !(is_first_chip || is_last_chip);

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

                uint32_t chunks_per_sync_val;
                if (normalized_dim == 0) {
                    chunks_per_sync_val =
                        chunks_per_sync.value_or(operations::experimental::ccl::detail::default_chunks_per_sync(
                            topology, start_tiles_to_read * slice_B, start_tiles_read * slice_B, tile_granularity));
                } else {
                    chunks_per_sync_val =
                        chunks_per_sync.value_or(operations::experimental::ccl::detail::default_chunks_per_sync(
                            topology, start_tiles_to_read * slice_C, start_tiles_read * slice_C, tile_granularity));
                }
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                // Reader CT args
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
                        is_forward,
                        is_first_device_in_direction,
                        num_targets_in_direction,
                        do_final_reduction,
                        sync_with_other_direction,
                        chunks_per_sync_val,
                        normalized_dim,
                        start_pages_read_in_row,
                        start_row_offset,
                        start_tiles_read,
                        start_tiles_to_read);
                if (input_is_sharded) {
                    shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
                } else {
                    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
                } else {
                    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer())
                        .append_to(sender_reader_compile_args);
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
                auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    sender_reader_kernel_path,
                    {core},
                    tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));
                reader_kernel_ids.push_back(worker_sender_reader_kernel_id);

                // Reader RT args
                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),         // input_tensor_address
                    intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                    output_tensor.buffer()->address(),        // output_tensor_address
                    semaphore.at(0).address(),                // remote transfer sync semaphore
                    fwd_bwd_semaphore_address};
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
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);
                CoreCoord termination_master_logical_core =
                    all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + 0];
                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                // Writer CT args
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
                        is_forward,
                        is_first_device_in_direction,
                        num_targets_in_direction,
                        do_final_reduction,
                        sync_with_other_direction,
                        chunks_per_sync_val,
                        normalized_dim,
                        start_pages_read_in_row,
                        start_row_offset,
                        start_tiles_read,
                        start_tiles_to_read);
                append_fabric_mux_connection_ct_args(
                    worker == 0,
                    mux_virtual_core,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    worker,
                    mux_kernel_config,
                    sender_writer_compile_args);
                if (is_forward) {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
                } else {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_writer_compile_args);
                } else {
                    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer())
                        .append_to(sender_writer_compile_args);
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
                auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    sender_writer_kernel_path,
                    {core},
                    tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));
                writer_kernel_ids.push_back(worker_sender_writer_kernel_id);
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
                        : 0};
                append_fabric_mux_connection_rt_args(
                    mux_connection_valid,
                    core,
                    program,
                    termination_master_virtual_core,
                    num_workers_per_direction,
                    writer_rt_args);
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, writer_rt_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
                }
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

                // Reduce CT args
                std::vector<uint32_t> sender_reduce_compile_args =
                    operations::experimental::ccl::detail::get_line_reduce_compile_args(
                        input_cb_index,
                        intermediate_cb_index,
                        compute_output_cb_index,
                        tile_granularity,
                        input_tensor_B,
                        slice_B,
                        slice_C,
                        num_total_reduction_steps,
                        start_tiles_read,
                        start_tiles_to_read,
                        normalized_dim);

                std::string sender_reduce_kernel_path =
                    normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                          "device/kernels/dim_zero_line_reduction.cpp"
                                        : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                          "device/kernels/line_reduction.cpp";
                auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    sender_reduce_kernel_path,
                    {core},
                    tt::tt_metal::ComputeConfig{.compile_args = sender_reduce_compile_args});
                reduce_kernel_ids.push_back(sender_reduce_kernel_id);

                // Reduce RT args
                std::vector<uint32_t> reduce_rt_args = {};
                tt::tt_metal::SetRuntimeArgs(program, sender_reduce_kernel_id, {core}, reduce_rt_args);
            }
        }
    }

    return {
        reader_kernel_ids,
        writer_kernel_ids,
        all_cores,
        num_directions_per_link,
        num_workers_per_direction,
        num_mux_cores_per_direction_per_link,
        num_cores_per_link};
}

void line_reduce_scatter_minimal_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const std::vector<tt::tt_metal::KernelHandle>& reader_kernel_ids,
    const std::vector<tt::tt_metal::KernelHandle>& writer_kernel_ids,
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
    uint32_t core_idx = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                uint32_t mux_core_offset = (link * num_cores_per_link) +
                                           (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                std::vector<std::vector<RuntimeArgsData>> reader_runtime_args =
                    GetRuntimeArgs(program, reader_kernel_ids[core_idx]);
                std::vector<std::vector<RuntimeArgsData>> writer_runtime_args =
                    GetRuntimeArgs(program, writer_kernel_ids[core_idx]);

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

                core_idx++;
            }
        }
    }
}

tt::tt_metal::operation::ProgramWithCallbacks line_reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
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
    auto
        [reader_kernel_ids,
         writer_kernel_ids,
         all_cores,
         num_directions_per_link,
         num_workers_per_direction,
         num_mux_cores_per_direction_per_link,
         num_cores_per_link] =
            build_line_reduce_scatter_minimal_async_program_artifacts(
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
    auto override_runtime_arguments_callback =
        [reader_kernel_ids,
         writer_kernel_ids,
         all_cores,
         num_links,
         num_directions_per_link,
         num_workers_per_direction,
         num_mux_cores_per_direction_per_link,
         num_cores_per_link](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[1];
            const auto& intermed = output_tensors[0];

            const auto& barrier_semaphore =
                static_cast<const ttnn::ReduceScatterMinimalAsync*>(operation)->barrier_semaphore;
            const auto& semaphore = static_cast<const ttnn::ReduceScatterMinimalAsync*>(operation)->semaphore;
            line_reduce_scatter_minimal_async_helper_override_runtime_arguments(
                program,
                reader_kernel_ids,
                writer_kernel_ids,
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
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}
}  // namespace ttnn
