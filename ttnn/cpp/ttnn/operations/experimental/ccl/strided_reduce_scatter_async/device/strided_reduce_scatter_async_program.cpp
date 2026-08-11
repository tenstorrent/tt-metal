// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

using namespace ccl;
using ttnn::experimental::ccl::append_fabric_mux_connection_ct_args;
using ttnn::experimental::ccl::append_fabric_mux_connection_rt_args;

/**
 * Strided Ring Reduce-Scatter
 *
 * Overview:
 *   A variant of bidirectional ring reduce-scatter where the input matrix is
 *   reduced in small portions called "chunks" rather than full slices. Each chunk
 *   corresponds to a strided block subrow of the minimal_matmul output. The purpose
 *   is to let the reduce-scatter fire as soon as the preceding matmul produces each
 *   chunk, overlapping communication with computation, rather than waiting for the
 *   full matmul to complete.
 *
 * Input / output:
 *   Each device holds one matrix of shape [B, 1, M, N] (one per batch element).
 *   The result is the element-wise sum of all devices' matrices, scattered along
 *   dim 3 (width) so each device keeps a reduced [B, 1, M, slice_Wt] portion
 *   where slice_Wt = N / ring_size.
 *
 * Data layout and naming (matmul output = RS input):
 *
 *   The matmul distributes work across a core grid (mm_cores_x × mm_cores_y).
 *   Each core computes a "full block" (N_full_block_wt wide, mm_block_ht * mm_M_unit_blocks_per_core
 *   tall in tiles), producing it in unit matmul blocks of (mm_block_ht × mm_block_wt)
 *   in block-row-major order. The full output matrix is tiled as:
 *
 *     ┌──────────────────── N (= ring_size × slice_Wt) ───────────────────┐
 *     │ N_full_block_wt  N_full_block_wt                                  │
 *     │◄───────────────►◄───────────────►                                 │
 *     │ ┌─────┬─────┐   ┌─────┬─────┐      ...   (mm_cores_x columns)     │
 *     │ │ u0  │ u1  │   │ u0  │ u1  │                                     │
 *     │ ├─────┼─────┤   ├─────┼─────┤   ◄── each unit matmul block is     │
 *     │ │ u2  │ u3  │   │ u2  │ u3  │       mm_block_ht × mm_block_wt     │
 *     │ ├─────┼─────┤   ├─────┼─────┤                                     │
 *     │ │ ... │ ... │   │ ... │ ... │       (mm_cores_y rows of full blks)│
 *     │ └─────┴─────┘   └─────┴─────┘                                     │
 *     │                                                                   │
 *     │ ◄── slice 0 ──► ◄── slice 1 ──►  ...  ◄── slice R-1 ──►           │
 *     └───────────────────────────────────────────────────────────────────┘
 *                             M rows (tile height)
 *
 *   Note: slice boundaries need not align with full block boundaries (see
 *   "Non-aligned case" below).
 *
 * Chunks:
 *   Each chunk is a strided block subrow: the collection of unit matmul blocks
 *   at the same column position across all MM cores. Chunk width is a
 *   hyperparameter (chunk_width_in_mm_blocks × mm_block_wt tiles). A chunk
 *   becomes ready for reduce-scatter the moment all matmul cores have finished
 *   writing the corresponding unit blocks.
 *
 * Chunk-based iteration (pseudocode):
 *
 *     for each batch b:
 *       for m_block_iter in 0 .. mm_M_unit_blocks_per_core-1:
 *         for chunk_idx in 0 .. chunks_per_mm_N_full_block-1:
 *           [wait for matmul to signal this chunk is ready]
 *           perform a full bidirectional ring reduce-scatter of the chunk
 *
 * Bidirectional ring reduce-scatter (per chunk):
 *   There are ring_size steps (i = 0 .. ring_size-1). In step 0, each device
 *   reads a slice of its own input. In steps 1..ring_size-1, each device
 *   receives partial results from its neighbor (via the intermediate buffer),
 *   reduces them with the local input slice, and forwards the result onward.
 *   The final step writes the fully reduced chunk to the output tensor.
 *
 *   Workers are split into forward and backward directions, with
 *   num_workers_per_direction workers in each. Within a chunk, tiles are
 *   partitioned across both directions using a striped scheme with stride
 *   2 * num_workers:
 *
 *     backward worker k  →  tiles k, k + 2*W, k + 4*W, ...
 *     forward  worker k  →  tiles k+W, k + 3*W, k + 5*W, ...   (W = num_workers)
 *
 *   Concretely, each group of 2*W consecutive tiles is split in half: the first
 *   W tiles go to backward workers 0..W-1, the next W tiles go to forward
 *   workers 0..W-1.
 *
 *   Workers with the same ID (and same direction) on different devices handle
 *   the same tile positions, so the ring reduce correctly accumulates partial
 *   results without any tile remapping between hops.
 *
 * Kernel workflow and circular buffer (CB) communication:
 *
 *   Each worker core runs three kernels: reader, compute, and writer.
 *
 *   Ring step 0 (no reduction needed):
 *     Reader ──[reader_output_cb]──► Writer
 *     The reader loads tiles from the input tensor directly into reader_output_cb.
 *     The writer consumes reader_output_cb and sends tiles to the intermediate
 *     buffer on the neighboring device via fabric, then signals the neighbor's
 *     reader with an "intermediate ready" semaphore increment.
 *
 *   Ring steps 1 .. ring_size-2 (intermediate reduction):
 *     Reader ──[input_cb]──────► Compute ──[output_cb]──► Writer
 *     Reader ──[intermediate_cb]──┘
 *     The reader waits for the "intermediate ready" semaphore from the previous
 *     device, then loads the local input slice into input_cb and the intermediate
 *     buffer into intermediate_cb. The compute kernel reduces them (add_tiles)
 *     and pushes the result into output_cb. The writer sends the result to the
 *     next device's intermediate buffer and signals "intermediate ready".
 *
 *   Ring step ring_size-1 (final reduction):
 *     Same as above, but the writer writes the reduced result to the local
 *     output tensor instead of sending it to the neighbor.
 *
 * Synchronization:
 *   Within a batch, cross-device synchronization is handled entirely by the
 *   "intermediate ready" semaphores — no costly global barriers are needed.
 *   Between batches, a barrier ensures all writers have finished before any
 *   device reuses its intermediate buffer (which holds only one batch element).
 *
 * Non-aligned cases:
 *   - If mm_cores_y does not divide slice_Ht, the last unit block per full block
 *     may contain "ghost tiles" (rows beyond slice_Ht). These are skipped via
 *     bounds checks in the reader/writer kernels.
 *   - If slice_Wt is not divisible by N_full_block_wt, a slice may straddle
 *     full block boundaries. The kernel iterates over all encapsulating full
 *     blocks but skips tiles outside the slice.
 *
 * Fusion signaling (MM -> RS):
 *   The matmul dataflow kernel responsible for writing output on each core
 *   (either dm_in0_sender.cpp or dm_in1_sender_out.cpp) increments a semaphore on the RS
 *   reader cores each time it finishes writing a unit block (after a cross-core
 *   synchronization delay to account for varying write latencies). The RS reader
 *   waits on this semaphore (noc_semaphore_wait_min) before processing each chunk.
 *
 *   RS -> next op: The RS writer can optionally signal a downstream op
 *   (fused_op_signaler) after completing its ring iterations.
 */
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
    std::optional<uint32_t> mm_window_blocks,
    std::optional<uint32_t> mm_logical_Ht,
    const std::optional<const Tensor>& mm_credit_counters,
    std::optional<float> fused_ternary_scalar,
    const std::optional<const Tensor>& addcmul_input_tensor1,
    const std::optional<const Tensor>& addcmul_input_tensor2,
    const std::optional<const Tensor>& mm_progress_counters) {
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
        num_workers_per_direction_opt.value_or(ttnn::experimental::ccl::reduce_scatter_default_workers(
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
    sender_worker_core_ranges.reserve(num_links * num_directions_per_link * num_workers_per_direction);
    std::vector<CoreRange> mux_core_ranges;
    mux_core_ranges.reserve(num_links * num_directions_per_link);
    std::vector<CoreRange> termination_master_core_ranges;
    termination_master_core_ranges.reserve(num_links * num_directions_per_link);
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
        (input_tensor_shape.rank() == 2)
            ? ttnn::experimental::ccl::reduce_scatter_map_2d_to_4d(dim)
            : ttnn::experimental::ccl::reduce_scatter_map_nd_to_4d(input_tensor_shape, dim);
    TT_FATAL(
        normalized_dim == 3,
        "strided_reduce_scatter_async ring implementation only supports scattering on dim 3 (width), but got {}",
        normalized_dim);
    // When the fused matmul hands its output over through a rolling L1 window, the input tensor is
    // only mm_window_blocks M blocks tall, so its height describes the window, not the data being
    // reduce-scattered. Take the true height from mm_logical_Ht in that case; the tensor is still the
    // right thing to build the TensorAccessor from, since the reader remaps rows into the window.
    const uint32_t input_tensor_Ht = mm_logical_Ht.value_or(input_tensor_shape[-2] / tt::constants::TILE_HEIGHT);
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

    // Page counts describe the logical reduce-scatter data, so they must not come from the buffer
    // when it only holds a window of it.
    const uint32_t input_tensor_num_pages = mm_logical_Ht.has_value()
                                                ? input_tensor_B * input_tensor_C * input_tensor_Ht * input_tensor_Wt
                                                : input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;
    const uint32_t input_batch_num_pages = input_tensor_num_pages / input_tensor_B;
    const uint32_t output_batch_num_pages = output_tensor_num_pages / slice_B;
    const uint32_t input_channel_num_pages = input_batch_num_pages / input_tensor_C;
    const uint32_t output_channel_num_pages = output_batch_num_pages / slice_C;

    // scatter-write supports up to 4 distinct noc addresses per packet
    uint32_t max_target_noc_addresses_per_packet = 4;

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

    [[maybe_unused]] bool input_is_sharded = input_tensor.is_sharded();  // input always via TensorAccessorArgs
    bool intermediate_is_sharded = intermediate_tensor.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;
    std::map<std::string, std::string> reduce_compute_defines;

    // The input (MM output) is always fed through TensorAccessorArgs (below)
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
        // Non-broadcast gate: b has full rows (per-token), element-wise multiply.
        // Broadcast gate: b has 1 row per tile, broadcast across acc's rows.
        auto b_logical_shape = addcmul_input_tensor2->logical_shape();
        if (b_logical_shape[-2] <= 1) {
            reader_compute_defines["ADDCMUL_B_BROADCAST"] = "1";
            reduce_compute_defines["ADDCMUL_B_BROADCAST"] = "1";
        }
    }

    // KERNEL CREATION
    std::vector<size_t> mux_termination_signal_addresses;
    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }
    bool fuse_mm_op = mm_fused_op_signaler.has_value();
    // Per-core MM signaling: L1 array of per-MM-core progress counters, one row per RS worker core
    std::shared_ptr<tt::tt_metal::Buffer> mm_progress_counters_buffer;
    uint32_t captured_mm_progress_counters_addr = 0;
    if (fuse_mm_op) {
        mm_fused_op_signaler->init_strided_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
        reader_compute_defines["FUSE_MM_OP_SIGNALER"] = "1";

        // The counter array: one shard (row) per RS worker core, sized to the full device compute grid
        const auto mm_grid = mesh_device->compute_with_storage_grid_size();
        const uint32_t num_mm_core_slots = mm_grid.x * mm_grid.y;
        const uint32_t counters_row_bytes = num_mm_core_slots * sizeof(uint32_t);

        if (mm_progress_counters.has_value()) {
            // Caller-owned array, shared across programs
            const auto& counters = mm_progress_counters.value();
            const auto& counters_shard_spec = counters.memory_config().shard_spec();
            TT_FATAL(
                counters.memory_config().buffer_type() == tt::tt_metal::BufferType::L1 &&
                    counters_shard_spec.has_value(),
                "mm_progress_counters must be an L1 sharded tensor so that its row lands at the same "
                "local address on every RS worker core");
            TT_FATAL(
                counters_shard_spec->grid.contains(sender_worker_core_range_set),
                "mm_progress_counters is sharded over {} which does not cover the RS worker cores {}",
                counters_shard_spec->grid.str(),
                sender_worker_core_range_set.str());
            const uint32_t provided_row_bytes = counters_shard_spec->shape[1] * counters.element_size();
            TT_FATAL(
                provided_row_bytes >= counters_row_bytes,
                "mm_progress_counters provides {} B per core but the MM grid needs {} slots ({} B)",
                provided_row_bytes,
                num_mm_core_slots,
                counters_row_bytes);
            mm_fused_op_signaler->mm_progress_counters_addr = static_cast<uint32_t>(counters.buffer()->address());
        } else {
            // BUILD-VERIFY: this is the ONE tt-metal buffer-API call to confirm against your tree
            const uint32_t num_rs_cores = sender_worker_core_range_set.num_cores();
            const auto counter_shard_spec = tt::tt_metal::ShardSpecBuffer(
                sender_worker_core_range_set,
                {1, num_mm_core_slots},
                tt::tt_metal::ShardOrientation::ROW_MAJOR,
                {1, num_mm_core_slots},
                {num_rs_cores, num_mm_core_slots});
            mm_progress_counters_buffer = tt::tt_metal::CreateBuffer(tt::tt_metal::ShardedBufferConfig{
                .device = mesh_device,
                .size = num_rs_cores * counters_row_bytes,
                .page_size = counters_row_bytes,
                .buffer_type = tt::tt_metal::BufferType::L1,
                .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                .shard_parameters = counter_shard_spec});
            mm_fused_op_signaler->mm_progress_counters_addr =
                static_cast<uint32_t>(mm_progress_counters_buffer->address());
        }
        captured_mm_progress_counters_addr = mm_fused_op_signaler->mm_progress_counters_addr;
    }

    // RS -> MM credits, the return path for the rolling window. One L1 row per MM core holding a
    // monotonic "M blocks consumed" counter per RS reader; the matmul waits on the MINIMUM across
    // readers before recycling a window slot. A single shared counter would not do: readers stripe
    // disjoint tiles and run at different rates, so a fast reader could satisfy a summed total while
    // a slow one still has the block to read. Each reader owns its slot and increments it once per M
    // block, so readers never contend for the same word.
    // HEIGHT_SHARDED over the MM grid => the row is at the same local L1 address on every MM core.
    // Signalling walks an explicit list of MM core NOC coords, mirroring how the matmul signals the
    // RS (OpSignaler::signal_op_per_core) rather than multicasting to a rectangle.
    std::shared_ptr<tt::tt_metal::Buffer> rs_credit_counters_buffer;
    const uint32_t num_rs_readers = num_directions_per_link * num_links * num_workers_per_direction;
    uint32_t rs_credit_counters_addr = 0;
    std::vector<CoreCoord> mm_cores_noc;
    uint32_t num_mm_cores = 0;
    uint32_t captured_rs_credit_counters_addr = 0;
    if (mm_window_blocks.has_value()) {
        TT_FATAL(fuse_mm_op, "mm_window_blocks requires the fused matmul path (FUSE_MM_OP_SIGNALER).");
        TT_FATAL(
            input_tensor_B == 1 && slice_C == 1,
            "mm_window_blocks currently assumes B=C=1; the window carries no batch/channel dimension "
            "(got B={}, C={}).",
            input_tensor_B,
            slice_C);

        const uint32_t mm_cores_x_val = tt::div_up(input_tensor_Wt, mm_N_full_block_wt_val);
        num_mm_cores = mm_cores_x_val * mm_cores_y_val;
        const CoreRangeSet mm_core_range_set(
            CoreRange(CoreCoord(0, 0), CoreCoord(mm_cores_x_val - 1, mm_cores_y_val - 1)));
        const uint32_t credit_row_bytes = num_rs_readers * sizeof(uint32_t);
        if (mm_credit_counters.has_value()) {
            // Caller-owned array, shared across programs. Same reasoning as mm_progress_counters:
            // a per-program copy is retained for as long as the program stays cached, and L1 is
            // handed out top-down, so each small permanent block pins the space above it.
            const auto& credits = mm_credit_counters.value();
            const auto& credits_shard_spec = credits.memory_config().shard_spec();
            TT_FATAL(
                credits.memory_config().buffer_type() == tt::tt_metal::BufferType::L1 && credits_shard_spec.has_value(),
                "mm_credit_counters must be an L1 sharded tensor so that its row lands at the same "
                "local address on every matmul core");
            TT_FATAL(
                credits_shard_spec->grid.contains(mm_core_range_set),
                "mm_credit_counters is sharded over {} which does not cover the matmul cores {}",
                credits_shard_spec->grid.str(),
                mm_core_range_set.str());
            const uint32_t provided_row_bytes = credits_shard_spec->shape[1] * credits.element_size();
            TT_FATAL(
                provided_row_bytes >= credit_row_bytes,
                "mm_credit_counters provides {} B per core but the {} RS readers need {} B",
                provided_row_bytes,
                num_rs_readers,
                credit_row_bytes);
            rs_credit_counters_addr = static_cast<uint32_t>(credits.buffer()->address());
        } else {
            const auto credit_shard_spec = tt::tt_metal::ShardSpecBuffer(
                mm_core_range_set,
                {1, num_rs_readers},
                tt::tt_metal::ShardOrientation::ROW_MAJOR,
                {1, num_rs_readers},
                {num_mm_cores, num_rs_readers});
            rs_credit_counters_buffer = tt::tt_metal::CreateBuffer(tt::tt_metal::ShardedBufferConfig{
                .device = mesh_device,
                .size = num_mm_cores * credit_row_bytes,
                .page_size = credit_row_bytes,
                .buffer_type = tt::tt_metal::BufferType::L1,
                .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                .shard_parameters = credit_shard_spec});
            rs_credit_counters_addr = static_cast<uint32_t>(rs_credit_counters_buffer->address());
        }

        // NOC coords of every MM core, in the same row-major order the matmul uses to index its own
        // slot, so a reader can walk them and bump its counter on each.
        mm_cores_noc.reserve(num_mm_cores);
        for (uint32_t y = 0; y < mm_cores_y_val; y++) {
            for (uint32_t x = 0; x < mm_cores_x_val; x++) {
                mm_cores_noc.push_back(mesh_device->worker_core_from_logical_core(CoreCoord(x, y)));
            }
        }

        // Hand the return path to the matmul factory, which builds its program after this one.
        mm_fused_op_signaler->rs_credit_counters_addr = rs_credit_counters_addr;
        mm_fused_op_signaler->num_rs_readers = num_rs_readers;
        mm_fused_op_signaler->mm_window_blocks = mm_window_blocks.value();
        captured_rs_credit_counters_addr = rs_credit_counters_addr;
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

    // CT arg indices must match kernel: see minimal_ring_strided_reduce_scatter_async_reader.cpp
    std::vector<uint32_t> sender_reader_compile_args = {
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
        mm_block_ht_val,                    // [15] mm_block_ht
        mm_cores_y_val,                     // [16] mm_cores_y
        mm_N_full_block_wt_val,             // [17] N_full_block_wt
        chunk_width_in_tiles_val,           // [18] chunk_width_in_tiles
        chunks_per_mm_N_full_block_val,     // [19] chunks_per_mm_N_full_block
        mm_block_wt_val,                    // [20] mm_block_wt (used by FUSE_MM_OP_SIGNALER)
        slice_Ht_per_core,                  // [21] slice_Ht_per_core
        static_cast<uint32_t>(fuse_mm_op),  // [22] fuse_mm_op (consumed via FUSE_MM_OP_SIGNALER define)
        slice_Ht,                           // [23] slice_Ht (total height in tiles across all MM cores)
        mm_window_blocks.value_or(0),       // [24] mm_window_blocks (0 = whole MM output resident)
    };

    // Input (MM output): always TensorAccessorArgs (handles interleaved AND L1-sharded) so the reader's
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
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
    // CT arg indices must match kernel: see minimal_ring_strided_reduce_scatter_async_writer.cpp
    // NOTE: writer does not receive fuse_mm_op; only reader needs to wait on the MM semaphore.
    std::vector<uint32_t> sender_writer_compile_args = {
        ring_index,                      // [0]  my_chip_id
        ring_size,                       // [1]  ring_size
        compute_output_cb_index,         // [2]  cb_compute_output_id
        reader_output_cb_index,          // [3]  cb_reader_output_id
        tile_granularity,                // [4]  packet_size_in_pages
        page_size,                       // [5]  page_size
        num_tiles_to_write_per_packet,   // [6]  num_tiles_to_write_per_packet
        output_batch_num_pages,          // [7]  output_batch_num_pages
        input_channel_num_pages,         // [8]  input_channel_num_pages
        output_channel_num_pages,        // [9]  output_channel_num_pages
        input_tensor_B,                  // [10] input_tensor_B
        input_tensor_Wt,                 // [11] input_tensor_Wt
        slice_C,                         // [12] slice_C
        slice_Wt,                        // [13] slice_Wt
        normalized_dim,                  // [14] dim normalized to 4D
        mm_M_unit_blocks_per_core,       // [15] mm_M_unit_blocks_per_core
        mm_block_ht_val,                 // [16] mm_block_ht
        mm_cores_y_val,                  // [17] mm_cores_y
        mm_N_full_block_wt_val,          // [18] N_full_block_wt
        chunk_width_in_tiles_val,        // [19] chunk_width_in_tiles
        chunks_per_mm_N_full_block_val,  // [20] chunks_per_mm_N_full_block
        slice_Ht_per_core,               // [21] slice_Ht_per_core
        slice_Ht,                        // [22] slice_Ht (unpadded; used for ghost-tile bounds checks)
        // [23+] fabric_mux CT args appended after (num_ct_args = 28 in writer kernel)
    };

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
    sender_reduce_kernel_config.compile_args = {
        input_cb_index,                  // [0]  input_cb_id
        intermediate_cb_index,           // [1]  intermediate_cb
        compute_output_cb_index,         // [2]  output_cb
        tile_granularity,                // [3]  tile_granularity
        ring_size,                       // [4]  ring_size
        input_tensor_B,                  // [5]  input_tensor_B
        mm_M_unit_blocks_per_core,       // [6]  mm_M_unit_blocks_per_core
        mm_block_ht_val,                 // [7]  mm_block_ht
        mm_cores_y_val,                  // [8]  mm_cores_y
        chunk_width_in_tiles_val,        // [9]  chunk_width_in_tiles
        chunks_per_mm_N_full_block_val,  // [10] chunks_per_mm_N_full_block
        slice_Wt,                        // [11] slice_Wt
        mm_N_full_block_wt_val,          // [12] mm_N_full_block_wt
        slice_Ht_per_core,               // [13] slice_Ht_per_core
        slice_Ht,                        // [14] slice_Ht (unpadded; used for ghost-tile bounds checks)
        ring_index,                      // [15] my_chip_id
    };
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
                // Input uses TensorAccessorArgs (see above) — no shard-map RT args, just the address.
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, reader_rt_args);
                }
                if (fuse_op) {
                    fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }
                if (fuse_mm_op) {
                    mm_fused_op_signaler->push_strided_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }
                if (mm_window_blocks.has_value()) {
                    // This reader's slot must match the kernel's effective_worker_id
                    // (worker_id + direction * num_workers), which is how tiles are striped.
                    reader_rt_args.push_back(rs_credit_counters_addr);
                    reader_rt_args.push_back(num_mm_cores);
                    for (const auto& c : mm_cores_noc) {
                        reader_rt_args.push_back(static_cast<uint32_t>(c.x));
                        reader_rt_args.push_back(static_cast<uint32_t>(c.y));
                    }
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
        captured_reader_addcmul_rt_arg_offset,
        mm_progress_counters_buffer,
        captured_mm_progress_counters_addr,
        rs_credit_counters_buffer,
        captured_rs_credit_counters_addr};
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
        std::nullopt,   // mm_window_blocks: standalone RS reads a full matmul output, never a window
        std::nullopt,   // mm_logical_Ht
        std::nullopt,   // mm_credit_counters
        std::nullopt,   // fused_ternary_scalar
        std::nullopt,   // addcmul_input_tensor1
        std::nullopt,   // addcmul_input_tensor2
        std::nullopt);  // mm_progress_counters (fused MM->RS only)

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
