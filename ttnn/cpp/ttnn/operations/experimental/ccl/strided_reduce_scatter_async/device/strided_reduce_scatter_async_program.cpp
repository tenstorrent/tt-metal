// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_program_utils.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_ring_program_factory.hpp"
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

#include <cstring>
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;
using namespace tt::tt_metal;

// Import types from the new TMP pattern
using ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail::StridedReduceScatterProgramArtifacts;

namespace ttnn {

namespace operations::experimental::ccl::strided_reduce_scatter_async::detail {

std::vector<uint32_t> get_ring_reader_compile_args(
    const uint32_t ring_index,
    const uint32_t ring_size,
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_C,
    const uint32_t slice_Wt,
    const uint32_t normalized_dim,
    const uint32_t mm_M_unit_blocks_per_core,
    const uint32_t mm_block_ht,
    const uint32_t mm_cores_y,
    const uint32_t N_full_block_wt,
    const uint32_t chunk_width_in_tiles,
    const uint32_t chunks_per_mm_N_full_block,
    const uint32_t mm_block_wt,
    const uint32_t slice_Ht_per_core,
    const bool fuse_mm_op,
    const uint32_t slice_Ht) {
    // Strided reader compile args - include MM blocking parameters
    // CT arg indices must match kernel: see minimal_ring_strided_reduce_scatter_async_reader.cpp
    return {
        ring_index,                         // [0]  my_chip_id
        ring_size,                          // [1]  ring_size
        input_cb_index,                     // [2]  cb_input_id
        intermediate_cb_index,              // [3]  cb_intermediate_id
        reader_output_cb_index,             // [4]  cb_reader_output_id
        tile_granularity,                   // [5]  tile_granularity
        page_size,                          // [6]  page_size
        input_batch_num_pages,              // [7]  input_batch_num_pages
        input_channel_num_pages,            // [8]  input_channel_num_pages
        input_tensor_B,                     // [9]  input_tensor_B
        input_tensor_Wt,                    // [10] input_tensor_Wt
        slice_C,                            // [11] slice_C
        slice_Wt,                           // [12] slice_Wt
        normalized_dim,                     // [13] dim normalized to 4D
        mm_M_unit_blocks_per_core,          // [14] mm_M_unit_blocks_per_core
        mm_block_ht,                        // [15] mm_block_ht
        mm_cores_y,                         // [16] mm_cores_y
        N_full_block_wt,                    // [17] N_full_block_wt
        chunk_width_in_tiles,               // [18] chunk_width_in_tiles
        chunks_per_mm_N_full_block,         // [19] chunks_per_mm_N_full_block
        mm_block_wt,                        // [20] mm_block_wt (used by FUSE_MM_OP_SIGNALER)
        slice_Ht_per_core,                  // [21] slice_Ht_per_core
        static_cast<uint32_t>(fuse_mm_op),  // [22] fuse_mm_op (consumed via FUSE_MM_OP_SIGNALER define)
        slice_Ht,                           // [23] slice_Ht (total height in tiles across all MM cores)
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
    const uint32_t output_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_C,
    const uint32_t slice_Wt,
    const uint32_t normalized_dim,
    const uint32_t mm_M_unit_blocks_per_core,
    const uint32_t mm_block_ht,
    const uint32_t mm_cores_y,
    const uint32_t N_full_block_wt,
    const uint32_t chunk_width_in_tiles,
    const uint32_t chunks_per_mm_N_full_block,
    const uint32_t slice_Ht_per_core,
    const uint32_t slice_Ht) {
    // Strided writer compile args - include MM blocking parameters
    // CT arg indices must match kernel: see minimal_ring_strided_reduce_scatter_async_writer.cpp
    // NOTE: writer does not receive fuse_mm_op; only reader needs to wait on the MM semaphore.
    return {
        ring_index,                     // [0]  my_chip_id
        ring_size,                      // [1]  ring_size
        compute_output_cb_index,        // [2]  cb_compute_output_id
        reader_output_cb_index,         // [3]  cb_reader_output_id
        tile_granularity,               // [4]  packet_size_in_pages
        page_size,                      // [5]  page_size
        num_tiles_to_write_per_packet,  // [6]  num_tiles_to_write_per_packet
        output_batch_num_pages,         // [7]  output_batch_num_pages
        input_channel_num_pages,        // [8]  input_channel_num_pages
        output_channel_num_pages,       // [9]  output_channel_num_pages
        input_tensor_B,                 // [10] input_tensor_B
        input_tensor_Wt,                // [11] input_tensor_Wt
        slice_C,                        // [12] slice_C
        slice_Wt,                       // [13] slice_Wt
        normalized_dim,                 // [14] dim normalized to 4D
        mm_M_unit_blocks_per_core,      // [15] mm_M_unit_blocks_per_core
        mm_block_ht,                    // [16] mm_block_ht
        mm_cores_y,                     // [17] mm_cores_y
        N_full_block_wt,                // [18] N_full_block_wt
        chunk_width_in_tiles,           // [19] chunk_width_in_tiles
        chunks_per_mm_N_full_block,     // [20] chunks_per_mm_N_full_block
        slice_Ht_per_core,              // [21] slice_Ht_per_core
        slice_Ht,                       // [22] slice_Ht (unpadded; used for ghost-tile bounds checks)
        // [23+] fabric_mux CT args appended after (num_ct_args = 28 in writer kernel)
    };
}

std::vector<uint32_t> get_ring_reduce_compile_args(
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t compute_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t ring_size,
    const uint32_t input_tensor_B,
    const uint32_t mm_M_unit_blocks_per_core,
    const uint32_t mm_block_ht,
    const uint32_t mm_cores_y,
    const uint32_t chunk_width_in_tiles,
    const uint32_t chunks_per_mm_N_full_block,
    const uint32_t slice_Wt,
    const uint32_t N_full_block_wt,
    const uint32_t slice_Ht_per_core,
    const uint32_t slice_Ht,
    const uint32_t my_chip_id) {
    // Strided reduction compile args - include MM blocking parameters
    return {
        input_cb_index,              // [0]  input_cb_id
        intermediate_cb_index,       // [1]  intermediate_cb
        compute_output_cb_index,     // [2]  output_cb
        tile_granularity,            // [3]  tile_granularity
        ring_size,                   // [4]  ring_size
        input_tensor_B,              // [5]  input_tensor_B
        mm_M_unit_blocks_per_core,   // [6]  mm_M_unit_blocks_per_core
        mm_block_ht,                 // [7]  mm_block_ht
        mm_cores_y,                  // [8]  mm_cores_y
        chunk_width_in_tiles,        // [9]  chunk_width_in_tiles
        chunks_per_mm_N_full_block,  // [10] chunks_per_mm_N_full_block
        slice_Wt,                    // [11] slice_Wt
        N_full_block_wt,             // [12] mm_N_full_block_wt
        slice_Ht_per_core,           // [13] slice_Ht_per_core
        slice_Ht,                    // [14] slice_Ht (unpadded; used for ghost-tile bounds checks)
        my_chip_id,                  // [15] my_chip_id
    };
}

}  // namespace operations::experimental::ccl::strided_reduce_scatter_async::detail

using namespace ccl;
using ttnn::experimental::ccl::append_fabric_mux_connection_ct_args;
using ttnn::experimental::ccl::append_fabric_mux_connection_rt_args;
namespace rs_detail = operations::experimental::ccl::strided_reduce_scatter_async::detail;

StridedReduceScatterProgramArtifacts build_ring_strided_reduce_scatter_async_program_artifacts(
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
    std::optional<experimental::ccl::StridedReduceScatterFusedOpSignaler>& mm_fused_op_signaler,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,
    std::optional<uint32_t> mm_cores_y,
    uint32_t mm_block_ht,
    uint32_t mm_block_wt,
    std::optional<uint32_t> mm_N_full_block_wt,
    std::optional<uint32_t> chunk_width_in_mm_blocks,
    std::optional<float> fused_ternary_scalar,
    const std::optional<const Tensor>& addcmul_input_tensor1,
    const std::optional<const Tensor>& addcmul_input_tensor2) {
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
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    uint32_t input_data_size_bytes = input_tensor.buffer()->size();
    uint32_t num_workers_per_direction =
        num_workers_per_direction_opt.value_or(ttnn::experimental::ccl::default_workers(
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

    uint32_t num_cores_per_link = ttnn::experimental::ccl::reduce_scatter_core_count_per_link(
        num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        sender_device_coord, forward_coord, backward_coord, mesh_device);
    auto [mcast_forward_args, mcast_backward_args] = ccl::get_forward_backward_line_mcast_configuration(
        sender_device_coord, forward_coord, backward_coord, ring_size - 1, ring_size - 1, mesh_device);

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
        (input_tensor_shape.rank() == 2) ? ttnn::experimental::ccl::map_2d_to_4d(dim)
                                         : ttnn::experimental::ccl::map_nd_to_4d(input_tensor_shape, dim);
    TT_FATAL(
        normalized_dim == 3,
        "strided_reduce_scatter_async ring implementation only supports scattering on dim 3 (width), but got {}",
        normalized_dim);
    const uint32_t input_tensor_Ht = input_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;

    const uint32_t slice_B = input_tensor_B;
    const uint32_t slice_C = input_tensor_C;
    const uint32_t slice_Ht = input_tensor_Ht;
    const uint32_t slice_Wt = input_tensor_Wt / ring_size;

    // MM blocking parameters
    const uint32_t mm_block_ht_val = mm_block_ht;
    const uint32_t mm_block_wt_val = mm_block_wt;
    const uint32_t mm_cores_y_val = mm_cores_y.value_or(tt::div_up(slice_Ht, mm_block_ht_val));
    const uint32_t mm_N_full_block_wt_val = mm_N_full_block_wt.value_or(slice_Wt);

    const uint32_t chunk_width_in_mm_blocks_val =
        chunk_width_in_mm_blocks.value_or(tt::div_up(mm_N_full_block_wt_val, mm_block_wt_val));
    const uint32_t chunk_width_in_tiles_val = chunk_width_in_mm_blocks_val * mm_block_wt_val;
    const uint32_t chunks_per_mm_N_full_block_val = tt::div_up(mm_N_full_block_wt_val, chunk_width_in_tiles_val);

    // Pad slice_Ht to the next multiple of mm_cores_y_val so every core gets an equal number of
    // tile rows. The last core may receive ghost tiles (slice_row >= slice_Ht) which are skipped
    // by the reader/writer kernels via bounds checks.
    const uint32_t padded_slice_Ht = tt::round_up(slice_Ht, mm_cores_y_val);
    const uint32_t slice_Ht_per_core = padded_slice_Ht / mm_cores_y_val;
    const uint32_t mm_M_unit_blocks_per_core = tt::div_up(slice_Ht_per_core, mm_block_ht_val);

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

    // Addcmul fused CBs (only created when fused_ternary_scalar is provided).
    // c_in4 = addcmul_temp (acc result before ternary ops), c_in5 = residual a, c_in6 = gate b.
    const bool fuse_rs_addcmul =
        fused_ternary_scalar.has_value() && addcmul_input_tensor1.has_value() && addcmul_input_tensor2.has_value();
    uint32_t addcmul_temp_cb_index = tt::CB::c_in4;
    uint32_t addcmul_a_cb_index = tt::CB::c_in5;
    uint32_t addcmul_b_cb_index = tt::CB::c_in6;
    if (fuse_rs_addcmul) {
        // Temp CB needs double capacity for the in-place mul-then-repack pattern.
        tt::tt_metal::CircularBufferConfig cb_addcmul_temp_config =
            tt::tt_metal::CircularBufferConfig(
                2 * cb_num_pages * l1_scratch_cb_page_size_bytes, {{addcmul_temp_cb_index, df}})
                .set_page_size(addcmul_temp_cb_index, l1_scratch_cb_page_size_bytes);
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_addcmul_temp_config);

        tt::tt_metal::CircularBufferConfig cb_addcmul_a_config =
            tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{addcmul_a_cb_index, df}})
                .set_page_size(addcmul_a_cb_index, l1_scratch_cb_page_size_bytes);
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_addcmul_a_config);

        tt::tt_metal::CircularBufferConfig cb_addcmul_b_config =
            tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{addcmul_b_cb_index, df}})
                .set_page_size(addcmul_b_cb_index, l1_scratch_cb_page_size_bytes);
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_addcmul_b_config);
    }

    bool input_is_sharded = input_tensor.is_sharded();
    bool intermediate_is_sharded = intermediate_tensor.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;
    std::map<std::string, std::string> reduce_compute_defines;

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
    if (fuse_rs_addcmul) {
        reader_compute_defines["FUSE_RS_ADDCMUL"] = "1";
        reduce_compute_defines["FUSE_RS_ADDCMUL"] = "1";
    }

    // KERNEL CREATION
    std::vector<size_t> mux_termination_signal_addresses;
    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }
    bool fuse_mm_op = mm_fused_op_signaler.has_value();
    if (fuse_mm_op) {
        mm_fused_op_signaler->init_strided_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
        reader_compute_defines["FUSE_MM_OP_SIGNALER"] = "1";
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

    std::vector<uint32_t> sender_reader_compile_args = rs_detail::get_ring_reader_compile_args(
        ring_index,
        ring_size,
        input_cb_index,
        intermediate_cb_index,
        reader_output_cb_index,
        tile_granularity,
        page_size,
        input_batch_num_pages,
        input_channel_num_pages,
        input_tensor_B,
        input_tensor_Wt,
        slice_C,
        slice_Wt,
        normalized_dim,
        mm_M_unit_blocks_per_core,
        mm_block_ht_val,
        mm_cores_y_val,
        mm_N_full_block_wt_val,
        chunk_width_in_tiles_val,
        chunks_per_mm_N_full_block_val,
        mm_block_wt_val,
        slice_Ht_per_core,
        fuse_mm_op,
        slice_Ht);

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
    // Addcmul tensor accessor CT args (a then b) — appended after intermediate.
    if (fuse_rs_addcmul) {
        sender_reader_compile_args.push_back(addcmul_a_cb_index);
        sender_reader_compile_args.push_back(addcmul_b_cb_index);
        tt::tt_metal::TensorAccessorArgs(addcmul_input_tensor1->buffer()).append_to(sender_reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(addcmul_input_tensor2->buffer()).append_to(sender_reader_compile_args);
    }

    std::string sender_reader_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/"
        "device/kernels/minimal_ring_strided_reduce_scatter_async_reader.cpp";

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_reader_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));

    // Writer
    std::vector<uint32_t> sender_writer_compile_args = rs_detail::get_ring_writer_compile_args(
        ring_index,
        ring_size,
        compute_output_cb_index,
        reader_output_cb_index,
        tile_granularity,
        page_size,
        num_tiles_to_write_per_packet,
        output_batch_num_pages,
        input_channel_num_pages,
        output_channel_num_pages,
        input_tensor_B,
        input_tensor_Wt,
        slice_C,
        slice_Wt,
        normalized_dim,
        mm_M_unit_blocks_per_core,
        mm_block_ht_val,
        mm_cores_y_val,
        mm_N_full_block_wt_val,
        chunk_width_in_tiles_val,
        chunks_per_mm_N_full_block_val,
        slice_Ht_per_core,
        slice_Ht);

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
        "ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/"
        "device/kernels/minimal_ring_strided_reduce_scatter_async_writer.cpp";

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_writer_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));

    // Reduce kernel
    auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
    sender_reduce_kernel_config.compile_args = rs_detail::get_ring_reduce_compile_args(
        input_cb_index,
        intermediate_cb_index,
        compute_output_cb_index,
        tile_granularity,
        ring_size,
        input_tensor_B,
        mm_M_unit_blocks_per_core,
        mm_block_ht_val,
        mm_cores_y_val,
        chunk_width_in_tiles_val,
        chunks_per_mm_N_full_block_val,
        slice_Wt,
        mm_N_full_block_wt_val,
        slice_Ht_per_core,
        slice_Ht,
        ring_index);
    // Append addcmul CB indices for the compute kernel.
    if (fuse_rs_addcmul) {
        sender_reduce_kernel_config.compile_args.push_back(addcmul_temp_cb_index);  // [16]
        sender_reduce_kernel_config.compile_args.push_back(addcmul_a_cb_index);     // [17]
        sender_reduce_kernel_config.compile_args.push_back(addcmul_b_cb_index);     // [18]
    }
    sender_reduce_kernel_config.defines = reduce_compute_defines;

    std::string sender_reduce_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/"
        "device/kernels/minimal_ring_reduction.cpp";

    auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
        program, sender_reduce_kernel_path, sender_worker_core_range_set, sender_reduce_kernel_config);

    // Captured from the first worker iteration; the same for all workers.
    uint32_t captured_reader_addcmul_rt_arg_offset = 0;

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

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),         // input_tensor_address
                    intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                    semaphore.at(dir).address(),              // out_ready_semaphore
                    dir,                                      // direction
                    worker_id,                                // worker_id
                    num_workers,                              // num_workers
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
                if (fuse_mm_op) {
                    mm_fused_op_signaler->push_strided_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }
                // Addcmul tensor addresses (a then b) — must be last so override_runtime_arguments
                // can locate them via reader_addcmul_rt_arg_offset.
                if (fuse_rs_addcmul) {
                    captured_reader_addcmul_rt_arg_offset = static_cast<uint32_t>(reader_rt_args.size());
                    reader_rt_args.push_back(addcmul_input_tensor1->buffer()->address());
                    reader_rt_args.push_back(addcmul_input_tensor2->buffer()->address());
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
                    dir,          // direction
                    worker_id,    // worker_id
                    num_workers,  // num_workers
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
                    dir,           // direction
                    worker_id,     // worker_id
                    num_workers};  // num_workers
                if (fuse_rs_addcmul) {
                    float scalar_f = fused_ternary_scalar.value();
                    uint32_t scalar_u32;
                    std::memcpy(&scalar_u32, &scalar_f, sizeof(uint32_t));
                    reduce_rt_args.push_back(scalar_u32);
                }
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
        num_cores_per_link,
        captured_reader_addcmul_rt_arg_offset};
}

void ring_strided_reduce_scatter_async_helper_override_runtime_arguments(
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
    const Tensor& output,
    uint32_t reader_addcmul_rt_arg_offset,
    const std::optional<const Tensor>& addcmul_a,
    const std::optional<const Tensor>& addcmul_b) {
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
                if (reader_addcmul_rt_arg_offset > 0 && addcmul_a.has_value() && addcmul_b.has_value()) {
                    worker_reader_sender_runtime_args[reader_addcmul_rt_arg_offset] = addcmul_a->buffer()->address();
                    worker_reader_sender_runtime_args[reader_addcmul_rt_arg_offset + 1] =
                        addcmul_b->buffer()->address();
                }
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

}  // namespace ttnn

// Implementations for the TMP namespace - wrappers to ttnn namespace functions
namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail {

// Mesh Workload Factory implementations
RingStridedReduceScatterMeshWorkloadFactory::cached_mesh_workload_t
RingStridedReduceScatterMeshWorkloadFactory::create_mesh_workload(
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

ttnn::device_operation::CachedProgram<RingStridedReduceScatterMeshWorkloadFactory::shared_variables_t>
RingStridedReduceScatterMeshWorkloadFactory::create_at(
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
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> mm_fused_op_signaler = std::nullopt;
    tt::tt_metal::Program program{};
    auto shared_vars = ::ttnn::build_ring_strided_reduce_scatter_async_program_artifacts(
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
        mm_fused_op_signaler,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        CoreCoord(0, 0),
        operation_attributes.mm_cores_y,
        operation_attributes.mm_block_ht,
        operation_attributes.mm_block_wt,
        operation_attributes.mm_N_full_block_wt,
        operation_attributes.chunk_width_in_mm_blocks,
        std::nullopt,   // fused_ternary_scalar
        std::nullopt,   // addcmul_input_tensor1
        std::nullopt);  // addcmul_input_tensor2

    return {std::move(program), std::move(shared_vars)};
}

void RingStridedReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& intermediate = tensor_return_value.at(0);
    const auto& output = tensor_return_value.at(1);

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        ::ttnn::ring_strided_reduce_scatter_async_helper_override_runtime_arguments(
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
            output,
            shared_vars.reader_addcmul_rt_arg_offset,
            std::nullopt,   // addcmul_a
            std::nullopt);  // addcmul_b
    }
}

}  // namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail
