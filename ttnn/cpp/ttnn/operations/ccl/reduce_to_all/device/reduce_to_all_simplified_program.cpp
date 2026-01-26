// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
// Simplified reduce_to_all program factory.
//
// KEY SIMPLIFICATIONS:
// 1. Only 2 dataflow kernels (reader + writer) instead of 4
// 2. All kernels run on shard cores only (no extra worker cores needed)
// 3. Full zero-copy for input, neighbor data, and output via CB aliasing
// 4. Single compute kernel handles both R1 and R2 reductions
//
// ARCHITECTURE:
// - Each shard core runs: reader, compute, writer
// - Reader: signals local input ready, waits for neighbor data
// - Compute: R1 reduction, then R2 reduction, then final normalization
// - Writer: sends local data to R1 neighbor, sends R1 result to R2 neighbor
//
// ZERO-COPY DATA PATHS:
// - Local input: CB aliased to input tensor shard (no memcpy on read)
// - R1 neighbor: CB aliased to R1 MeshBuffer (direct fabric write)
// - R2 neighbor: CB aliased to R2 MeshBuffer (direct fabric write)
// - Final output: CB aliased to output tensor shard (no memcpy on write)

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "reduce_to_all_op.hpp"

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::ccl {
namespace {

// Helper to add mux compile-time args (prefixed to avoid unity build conflicts)
void simplified_fabric_mux_ct_args(
    const uint32_t num_workers_per_direction,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args) {
    worker_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));
    worker_ct_args.push_back(mux_kernel_config.get_buffer_size_bytes(channel_type));
    worker_ct_args.push_back(mux_kernel_config.get_status_address());
    worker_ct_args.push_back(mux_kernel_config.get_termination_signal_address());
    worker_ct_args.push_back(num_workers_per_direction);
}

// Helper to add mux runtime args (prefixed to avoid unity build conflicts)
void simplified_fabric_mux_rt_args(
    const bool is_termination_master,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    const uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    uint32_t shared_termination_sync_sem,
    std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(is_termination_master);
    worker_rt_args.push_back(mux_virtual_core.x);
    worker_rt_args.push_back(mux_virtual_core.y);
    worker_rt_args.push_back(mux_kernel_config.get_channel_base_address(channel_type, worker_id));
    worker_rt_args.push_back(mux_kernel_config.get_connection_info_address(channel_type, worker_id));
    worker_rt_args.push_back(mux_kernel_config.get_connection_handshake_address(channel_type, worker_id));
    worker_rt_args.push_back(mux_kernel_config.get_flow_control_address(channel_type, worker_id));
    worker_rt_args.push_back(mux_kernel_config.get_buffer_index_address(channel_type, worker_id));
    worker_rt_args.push_back(mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));
    worker_rt_args.push_back(shared_termination_sync_sem);
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(termination_master_virtual_core.x);
    worker_rt_args.push_back(termination_master_virtual_core.y);
}

}  // anonymous namespace

ttnn::device_operation::CachedProgram<ReduceToAllOp::ReduceToAll::shared_variables_t>
reduce_to_all_simplified_program_factory(
    const ReduceToAllOp::tensor_args_t& tensor_args,
    const ReduceToAllOp::operation_attributes_t& operation_attributes,
    [[maybe_unused]] const MeshCoordinate& root_coord,
    const float scale_fp32,
    const MeshCoordinate& device_coordinate,
    std::optional<ttnn::MeshCoordinate>& forward_coord,
    std::optional<ttnn::MeshCoordinate>& backward_coord,
    ReduceToAllOp::tensor_return_value_t& output_tensors,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    // =========================================================================
    // Setup
    // =========================================================================
    auto* mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor_l.device());
    auto* device = tensor_args.input_tensor_l.device();

    if (forward_coord.has_value()) {
        log_info(
            tt::LogTest,
            "physical device id: {}, current coord: {}, forward coordinate: {}",
            mesh_device->get_device(device_coordinate)->id(),
            device_coordinate,
            forward_coord.value());
    }
    if (backward_coord.has_value()) {
        log_info(
            tt::LogTest,
            "physical device id: {}, current coord: {}, backward coordinate: {}",
            mesh_device->get_device(device_coordinate)->id(),
            device_coordinate,
            backward_coord.value());
    }

    // Determine which mux direction to use for R1 and R2 based on device position.
    // Ring topology: D0 ↔ D1 ↔ D2 ↔ D3 ↔ (back to D0)
    //
    // Round 1 pairs: (D0,D1) and (D2,D3) exchange
    // Round 2 pairs: (D1,D2) and (D3,D0) exchange
    //
    // Direction mapping:
    //   D0: R1 sends FWD→D1,  R2 sends BWD→D3
    //   D1: R1 sends BWD→D0,  R2 sends FWD→D2
    //   D2: R1 sends FWD→D3,  R2 sends BWD→D1
    //   D3: R1 sends BWD→D2,  R2 sends FWD→D0
    //
    // Pattern: Even devices (0,2): R1=FWD, R2=BWD
    //          Odd devices (1,3):  R1=BWD, R2=FWD
    const uint32_t device_index = device_coordinate[0];  // Assuming 1D ring on first dimension
    const bool is_even_device = (device_index % 2) == 0;
    // For even devices: R1 uses FWD mux, R2 uses BWD mux
    // For odd devices:  R1 uses BWD mux, R2 uses FWD mux
    const bool r1_uses_fwd_mux = is_even_device;

    const auto& input_tensor_l = tensor_args.input_tensor_l;
    const auto& input_tensor_s = tensor_args.input_tensor_s;
    const auto& input_tensor_m = tensor_args.input_tensor_m;

    const auto& output_tensor_l = output_tensors.at(1)[0];
    const auto& output_tensor_s = output_tensors.at(1)[1];
    const auto& output_tensor_m = output_tensors.at(1)[2];

    // Use intermediate tensors as receive buffers.
    // These are MeshDevice tensors created ONCE at the mesh level, so they have
    // the SAME L1 address on ALL devices. This is critical for fabric sends!
    //
    // For the simplified 2-kernel design:
    // - R1 receive buffer: Use fw_intermediate_tensor (for R1 neighbor data)
    // - R2 receive buffer: Use bw_intermediate_tensor (for R2 neighbor data)
    //
    // The original design uses these tensors differently across device types,
    // but the key insight is that fw_intermediate and bw_intermediate have the
    // same address on all devices, allowing cross-device sends to work.
    const auto& r1_recv_tensor = output_tensors.at(0)[0];  // fw_intermediate
    const auto& r2_recv_tensor = output_tensors.at(0)[1];  // bw_intermediate

    // =========================================================================
    // Extract shard info and compute grid
    // =========================================================================
    TT_FATAL(input_tensor_l.is_sharded(), "Input tensor must be sharded");
    const auto& shard_spec = input_tensor_l.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;

    std::vector<CoreCoord> shard_cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto cores = corerange_to_cores(core_range, std::nullopt);
        shard_cores.insert(shard_cores.end(), cores.begin(), cores.end());
    }
    const uint32_t num_shard_cores = shard_cores.size();

    // =========================================================================
    // Compute parameters
    // =========================================================================
    const uint32_t input_page_size_bytes = input_tensor_l.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t aligned_page_size = tt::round_up(input_page_size_bytes, l1_alignment);

    const uint32_t input_l_total_num_pages = data_movement::get_num_pages(input_tensor_l);
    const uint32_t input_l_num_pages = input_l_total_num_pages / num_shard_cores;

    // SDPA compute dimensions
    const auto tile_width = input_tensor_l.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor_l.tensor_spec().tile().get_height();
    const uint32_t PNH = 8;
    const uint32_t DH = input_l_num_pages * tile_width;
    const uint32_t DHt = DH / tile_width;
    const uint32_t vDHt = DHt;  // For SDPA
    const uint32_t PNHt = PNH / tile_height;
    const uint32_t Sq_chunk_t = PNHt;
    const uint32_t out_tiles = Sq_chunk_t * vDHt;

    const uint32_t payload_size_bytes = out_tiles * input_page_size_bytes;
    const uint32_t total_packet_size = payload_size_bytes + 2 * aligned_page_size;  // L + S + M

    // =========================================================================
    // Use intermediate tensors as receive buffers
    // =========================================================================
    // The intermediate tensors (r1_recv_tensor, r2_recv_tensor) are MeshDevice tensors
    // created ONCE at the mesh level via create_output_tensors(). They have the SAME
    // L1 buffer address on ALL devices, which is critical for fabric sends.
    //
    // This replaces the per-device MeshBuffer::create() approach which was broken
    // because it created different buffer instances (with different addresses) for
    // each device.

    // Scale encoding
    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = scale_fp32;
    uint32_t scale_val = scale_union.u;

    // =========================================================================
    // Create Program
    // =========================================================================
    tt::tt_metal::Program program{};

    const auto tiny_tile = tt::tt_metal::Tile({8, 32});
    auto stats_tile = tiny_tile;
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_l.dtype());

    // =========================================================================
    // Define CB indices
    // =========================================================================
    // R1 local input (aliased to input tensor shard)
    constexpr auto cb_local_l = tt::CBIndex::c_0;
    constexpr auto cb_local_s = tt::CBIndex::c_1;
    constexpr auto cb_local_m = tt::CBIndex::c_2;

    // R1 neighbor input (aliased to R1 MeshBuffer)
    constexpr auto cb_r1_neighbor_l = tt::CBIndex::c_3;
    constexpr auto cb_r1_neighbor_s = tt::CBIndex::c_4;
    constexpr auto cb_r1_neighbor_m = tt::CBIndex::c_5;

    // R1 result / R2 local input (writer sends to R2 neighbor)
    constexpr auto cb_r1_result_l = tt::CBIndex::c_6;
    constexpr auto cb_r1_result_s = tt::CBIndex::c_7;
    constexpr auto cb_r1_result_m = tt::CBIndex::c_8;

    // R2 neighbor input (aliased to R2 MeshBuffer)
    constexpr auto cb_r2_neighbor_l = tt::CBIndex::c_9;
    constexpr auto cb_r2_neighbor_s = tt::CBIndex::c_10;
    constexpr auto cb_r2_neighbor_m = tt::CBIndex::c_11;

    // Temp CBs for compute (some are ALIASED to output tensor shards!)
    constexpr auto cb_exp_p1 = tt::CBIndex::c_12;
    constexpr auto cb_exp_p2 = tt::CBIndex::c_13;
    constexpr auto cb_m_out = tt::CBIndex::c_14;    // ALIASED to output_m
    constexpr auto cb_s1_temp = tt::CBIndex::c_15;  // intermediate only
    constexpr auto cb_s2_temp = tt::CBIndex::c_16;  // intermediate only
    constexpr auto cb_l_out = tt::CBIndex::c_17;    // ALIASED to output_l
    constexpr auto cb_l2_temp = tt::CBIndex::c_18;  // intermediate only
    constexpr auto cb_s_out = tt::CBIndex::c_19;    // ALIASED to output_s

    // Packet CBs for writer
    constexpr auto cb_packet_header = tt::CBIndex::c_20;
    constexpr auto cb_packet = tt::CBIndex::c_21;

    // Sync CB for coordination between Compute and Writer
    constexpr auto cb_sync = tt::CBIndex::c_22;

    // =========================================================================
    // Create Circular Buffers
    // =========================================================================
    // Local input CBs - ALIASED to input tensor shards (zero-copy reads)
    tt::tt_metal::CircularBufferConfig cb_local_l_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_local_l, input_dataformat}})
            .set_page_size(cb_local_l, aligned_page_size)
            .set_tile_dims(cb_local_l, stats_tile)
            .set_globally_allocated_address(*input_tensor_l.buffer());
    CreateCircularBuffer(program, shard_grid, cb_local_l_config);

    tt::tt_metal::CircularBufferConfig cb_local_s_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_local_s, input_dataformat}})
            .set_page_size(cb_local_s, aligned_page_size)
            .set_tile_dims(cb_local_s, stats_tile)
            .set_globally_allocated_address(*input_tensor_s.buffer());
    CreateCircularBuffer(program, shard_grid, cb_local_s_config);

    tt::tt_metal::CircularBufferConfig cb_local_m_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_local_m, input_dataformat}})
            .set_page_size(cb_local_m, aligned_page_size)
            .set_tile_dims(cb_local_m, stats_tile)
            .set_globally_allocated_address(*input_tensor_m.buffer());
    CreateCircularBuffer(program, shard_grid, cb_local_m_config);

    // R1 neighbor CBs - ALIASED to R1 receive tensor (direct fabric write, zero-copy receive)
    tt::tt_metal::CircularBufferConfig cb_r1_neighbor_l_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_r1_neighbor_l, input_dataformat}})
            .set_page_size(cb_r1_neighbor_l, aligned_page_size)
            .set_tile_dims(cb_r1_neighbor_l, stats_tile)
            .set_globally_allocated_address(*r1_recv_tensor.buffer());
    CreateCircularBuffer(program, shard_grid, cb_r1_neighbor_l_config);

    // S and M CBs are NOT aliased - reader will memcpy from buffer after packet arrives
    // (CB aliasing doesn't support offsets, and S/M are tiny - ~256 bytes each)
    tt::tt_metal::CircularBufferConfig cb_r1_neighbor_s_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_r1_neighbor_s, input_dataformat}})
            .set_page_size(cb_r1_neighbor_s, aligned_page_size)
            .set_tile_dims(cb_r1_neighbor_s, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r1_neighbor_s_config);

    tt::tt_metal::CircularBufferConfig cb_r1_neighbor_m_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_r1_neighbor_m, input_dataformat}})
            .set_page_size(cb_r1_neighbor_m, aligned_page_size)
            .set_tile_dims(cb_r1_neighbor_m, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r1_neighbor_m_config);

    // R1 result CBs (intermediate, not aliased)
    tt::tt_metal::CircularBufferConfig cb_r1_result_l_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_r1_result_l, input_dataformat}})
            .set_page_size(cb_r1_result_l, aligned_page_size)
            .set_tile_dims(cb_r1_result_l, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r1_result_l_config);

    tt::tt_metal::CircularBufferConfig cb_r1_result_s_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_r1_result_s, input_dataformat}})
            .set_page_size(cb_r1_result_s, aligned_page_size)
            .set_tile_dims(cb_r1_result_s, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r1_result_s_config);

    tt::tt_metal::CircularBufferConfig cb_r1_result_m_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_r1_result_m, input_dataformat}})
            .set_page_size(cb_r1_result_m, aligned_page_size)
            .set_tile_dims(cb_r1_result_m, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r1_result_m_config);

    // R2 neighbor CBs - ALIASED to R2 receive tensor (direct fabric write, zero-copy receive)
    tt::tt_metal::CircularBufferConfig cb_r2_neighbor_l_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_r2_neighbor_l, input_dataformat}})
            .set_page_size(cb_r2_neighbor_l, aligned_page_size)
            .set_tile_dims(cb_r2_neighbor_l, stats_tile)
            .set_globally_allocated_address(*r2_recv_tensor.buffer());
    CreateCircularBuffer(program, shard_grid, cb_r2_neighbor_l_config);

    tt::tt_metal::CircularBufferConfig cb_r2_neighbor_s_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_r2_neighbor_s, input_dataformat}})
            .set_page_size(cb_r2_neighbor_s, aligned_page_size)
            .set_tile_dims(cb_r2_neighbor_s, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r2_neighbor_s_config);

    tt::tt_metal::CircularBufferConfig cb_r2_neighbor_m_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_r2_neighbor_m, input_dataformat}})
            .set_page_size(cb_r2_neighbor_m, aligned_page_size)
            .set_tile_dims(cb_r2_neighbor_m, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r2_neighbor_m_config);

    // Temp / Output CBs for compute
    // cb_l_out, cb_s_out, cb_m_out are ALIASED to output tensor shards!
    tt::tt_metal::CircularBufferConfig cb_exp_p1_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_exp_p1, input_dataformat}})
            .set_page_size(cb_exp_p1, aligned_page_size)
            .set_tile_dims(cb_exp_p1, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_exp_p1_config);

    tt::tt_metal::CircularBufferConfig cb_exp_p2_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_exp_p2, input_dataformat}})
            .set_page_size(cb_exp_p2, aligned_page_size)
            .set_tile_dims(cb_exp_p2, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_exp_p2_config);

    // cb_m_out is ALIASED to output_tensor_m shard - compute writes directly to output!
    tt::tt_metal::CircularBufferConfig cb_m_out_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_m_out, input_dataformat}})
            .set_page_size(cb_m_out, aligned_page_size)
            .set_tile_dims(cb_m_out, stats_tile)
            .set_globally_allocated_address(*output_tensor_m.buffer());
    CreateCircularBuffer(program, shard_grid, cb_m_out_config);

    tt::tt_metal::CircularBufferConfig cb_s1_temp_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_s1_temp, input_dataformat}})
            .set_page_size(cb_s1_temp, aligned_page_size)
            .set_tile_dims(cb_s1_temp, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_s1_temp_config);

    tt::tt_metal::CircularBufferConfig cb_s2_temp_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_s2_temp, input_dataformat}})
            .set_page_size(cb_s2_temp, aligned_page_size)
            .set_tile_dims(cb_s2_temp, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_s2_temp_config);

    // cb_l_out is ALIASED to output_tensor_l shard - compute writes directly to output!
    tt::tt_metal::CircularBufferConfig cb_l_out_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_l_out, input_dataformat}})
            .set_page_size(cb_l_out, aligned_page_size)
            .set_tile_dims(cb_l_out, stats_tile)
            .set_globally_allocated_address(*output_tensor_l.buffer());
    CreateCircularBuffer(program, shard_grid, cb_l_out_config);

    tt::tt_metal::CircularBufferConfig cb_l2_temp_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_l2_temp, input_dataformat}})
            .set_page_size(cb_l2_temp, aligned_page_size)
            .set_tile_dims(cb_l2_temp, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_l2_temp_config);

    // cb_s_out is ALIASED to output_tensor_s shard - compute writes directly to output!
    tt::tt_metal::CircularBufferConfig cb_s_out_config =
        tt::tt_metal::CircularBufferConfig(Sq_chunk_t * aligned_page_size, {{cb_s_out, input_dataformat}})
            .set_page_size(cb_s_out, aligned_page_size)
            .set_tile_dims(cb_s_out, stats_tile)
            .set_globally_allocated_address(*output_tensor_s.buffer());
    CreateCircularBuffer(program, shard_grid, cb_s_out_config);

    // Packet CBs for writer
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            4 * packet_header_size_bytes, {{cb_packet_header, tt::DataFormat::RawUInt32}})
            .set_page_size(cb_packet_header, packet_header_size_bytes)
            .set_tile_dims(cb_packet_header, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_packet_header_config);

    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(2 * total_packet_size, {{cb_packet, input_dataformat}})
            .set_page_size(cb_packet, total_packet_size)
            .set_tile_dims(cb_packet, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_packet_config);

    // Sync CB (tiny, 1 tile size, can use 16B if supported but using aligned_page_size for safety)
    tt::tt_metal::CircularBufferConfig cb_sync_config =
        tt::tt_metal::CircularBufferConfig(aligned_page_size, {{cb_sync, tt::DataFormat::RawUInt32}})
            .set_page_size(cb_sync, aligned_page_size)
            .set_tile_dims(cb_sync, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_sync_config);

    // =========================================================================
    // Setup relay masters and relay buffers
    // =========================================================================
    // Relay optimization: Only 1 worker per link sends to mux (relay master).
    // Other 3 workers (relay workers) send their packets to the relay master via NOC.
    // Relay master: first core per link
    // Relay workers: remaining 3 cores per link
    constexpr auto num_links = 2;
    const uint32_t cores_per_link = num_shard_cores / num_links;
    constexpr uint32_t num_relay_workers = 3;  // Workers 1, 2, 3 relay through worker 0

    // Identify relay master cores (first core of each link)
    std::vector<CoreCoord> relay_master_cores = {shard_cores[0], shard_cores[cores_per_link]};

    // Relay buffer size: header + payload (L + S + M aligned)
    // Note: packet_header_size_bytes is already defined above for CB creation
    const uint32_t relay_buffer_size = tt::round_up(packet_header_size_bytes + total_packet_size, l1_alignment);

    // Create relay semaphores on relay master cores (one per relay worker)
    // These semaphores signal when a relay worker has written its packet to the relay buffer
    // Reused for both R1 and R2 phases
    std::vector<std::vector<uint32_t>> relay_sems_per_link(num_links);
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        CoreCoord relay_master = relay_master_cores[link_idx];
        for (uint32_t i = 0; i < num_relay_workers; i++) {
            relay_sems_per_link[link_idx].push_back(CreateSemaphore(program, {relay_master}, 0));
        }
    }

    // Allocate relay buffers on relay master cores using L1 space
    // We use a fixed offset from the base L1 address to avoid CB allocation complexity
    // The relay buffers are placed after the CB space
    const uint32_t l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // Estimate CB space usage (conservative upper bound)
    // This ensures relay buffers don't overlap with CB allocations
    const uint32_t estimated_cb_space =
        (out_tiles * aligned_page_size) * 6 +  // L tensors: local, r1_neighbor, r1_result, r2_neighbor, l_out, l2_temp
        (Sq_chunk_t * aligned_page_size) * 12 +  // S/M tensors and temps
        (4 * packet_header_size_bytes) +         // packet_header CB
        (2 * total_packet_size) +                // packet CB
        aligned_page_size +                      // sync CB
        4096;                                    // Safety margin

    // Relay buffer base address (after CB space)
    const uint32_t relay_buffer_base = l1_unreserved_base + estimated_cb_space;

    // Calculate relay buffer addresses for each link
    // Each relay master has 3 relay buffers (one per relay worker)
    std::vector<std::vector<uint32_t>> relay_buffer_addrs_per_link(num_links);
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        for (uint32_t i = 0; i < num_relay_workers; i++) {
            relay_buffer_addrs_per_link[link_idx].push_back(relay_buffer_base + i * relay_buffer_size);
        }
    }

    // =========================================================================
    // Setup mux cores and config
    // =========================================================================
    // With relay optimization: only 1 client per mux (the relay master)
    // Previously: num_workers_per_direction = num_shard_cores / num_links = 4
    constexpr uint32_t num_mux_clients_per_direction = 1;  // Only relay master connects to mux

    std::vector<CoreCoord> mux_cores = {CoreCoord(2, 0), CoreCoord(2, 1), CoreCoord(2, 2), CoreCoord(2, 3)};
    if (operation_attributes.input_mux_cores.has_value()) {
        mux_cores = operation_attributes.input_mux_cores.value();
    }
    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_cores);

    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    tt::tt_fabric::FabricMuxConfig mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_mux_clients_per_direction, 0, 2, 0, buffer_size_bytes_full_size_channel, mux_base_l1_address);

    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    // =========================================================================
    // Create kernel compile-time args
    // =========================================================================

    // Reader compile-time args
    // Note: cb_r1_neighbor_l and cb_r2_neighbor_l are aliased to receive buffers (zero-copy for L)
    // But S and M need memcpy from buffer since we can't alias at offsets
    const uint32_t s_tile_size_bytes = Sq_chunk_t * aligned_page_size;  // Size of S data
    const uint32_t m_tile_size_bytes = Sq_chunk_t * aligned_page_size;  // Size of M data
    std::vector<uint32_t> reader_ct_args = {
        Sq_chunk_t,             // 0
        vDHt,                   // 1
        input_page_size_bytes,  // 2
        l1_alignment,           // 3
        cb_local_l,             // 4
        cb_local_s,             // 5
        cb_local_m,             // 6
        cb_r1_neighbor_l,       // 7
        cb_r1_neighbor_s,       // 8
        cb_r1_neighbor_m,       // 9
        cb_r2_neighbor_l,       // 10
        cb_r2_neighbor_s,       // 11
        cb_r2_neighbor_m,       // 12
        payload_size_bytes,     // 13 - offset to S data in receive buffer
        s_tile_size_bytes,      // 14 - size of S data to copy
        m_tile_size_bytes,      // 15 - size of M data to copy
    };

    // Writer compile-time args
    // Build base args first, then calculate mux indices based on size
    std::vector<uint32_t> writer_ct_args = {
        Sq_chunk_t,                          // 0
        vDHt,                                // 1
        cb_local_l,                          // 2
        cb_local_s,                          // 3
        cb_local_m,                          // 4
        cb_r1_result_l,                      // 5
        cb_r1_result_s,                      // 6
        cb_r1_result_m,                      // 7
        cb_packet_header,                    // 8
        cb_packet,                           // 9
        l1_alignment,                        // 10
        input_page_size_bytes,               // 11
        cb_sync,                             // 12
        num_relay_workers,                   // 13 - number of relay workers (3)
        relay_buffer_size,                   // 14 - size of each relay buffer
        (uint32_t)packet_header_size_bytes,  // 15 - packet header size
    };

    // Calculate mux indices based on current size.
    // The indices will be at positions 16 and 17, followed by the mux args.
    // r1_mux_ct_idx points to where R1 mux args start (after base args + 2 indices)
    // r2_mux_ct_idx points to where R2 mux args start (after R1 mux args)
    constexpr uint32_t num_mux_args_per_direction = 5;
    uint32_t r1_mux_ct_idx = writer_ct_args.size() + 2;  // +2 for the two index args we're about to add
    uint32_t r2_mux_ct_idx = r1_mux_ct_idx + num_mux_args_per_direction;

    // Push the calculated indices (positions 16 and 17)
    writer_ct_args.push_back(r1_mux_ct_idx);
    writer_ct_args.push_back(r2_mux_ct_idx);

    // Add mux compile-time args for R1 and R2.
    // NOTE: The compile-time args are IDENTICAL for both muxes because they come from
    // the same mux_kernel_config. The FWD vs BWD direction only affects RUNTIME args
    // (mux core coordinates, semaphores). The memory layout, buffer sizes, status
    // addresses, etc. are the same for all muxes sharing a config.
    // With relay optimization, only 1 client (relay master) connects to each mux.
    simplified_fabric_mux_ct_args(
        num_mux_clients_per_direction,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        writer_ct_args);  // R1 mux config
    simplified_fabric_mux_ct_args(
        num_mux_clients_per_direction,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        writer_ct_args);  // R2 mux config (identical values, different runtime mapping)

    // Compute compile-time args (23 total)
    // Layout matches compute.cpp expectations:
    // - 0-2: Local input CBs
    // - 3-5: R1 neighbor CBs
    // - 6-8: R1 result CBs (sent to R2 neighbor by writer)
    // - 9-11: R2 neighbor CBs
    // - 12-13: Exponent temp CBs (P1, P2)
    // - 14: cb_m_out (ALIASED to output_tensor_m shard - no move needed!)
    // - 15-16: s temp CBs
    // - 17: cb_l_out (ALIASED to output_tensor_l shard - no move needed!)
    // - 18: l2_temp
    // - 19: cb_s_out (ALIASED to output_tensor_s shard - no move needed!)
    // - 20-22: scale, Sq_chunk_t, vDHt
    std::vector<uint32_t> compute_ct_args = {
        cb_local_l,        // 0
        cb_local_s,        // 1
        cb_local_m,        // 2
        cb_r1_neighbor_l,  // 3
        cb_r1_neighbor_s,  // 4
        cb_r1_neighbor_m,  // 5
        cb_r1_result_l,    // 6
        cb_r1_result_s,    // 7
        cb_r1_result_m,    // 8
        cb_r2_neighbor_l,  // 9
        cb_r2_neighbor_s,  // 10
        cb_r2_neighbor_m,  // 11
        cb_exp_p1,         // 12
        cb_exp_p2,         // 13
        cb_m_out,          // 14 - ALIASED to output_m
        cb_s1_temp,        // 15
        cb_s2_temp,        // 16
        cb_l_out,          // 17 - ALIASED to output_l
        cb_l2_temp,        // 18
        cb_s_out,          // 19 - ALIASED to output_s
        scale_val,         // 20
        Sq_chunk_t,        // 21
        vDHt,              // 22
        cb_sync,           // 23
    };

    // =========================================================================
    // Create kernels
    // =========================================================================
    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/reader.cpp",
        shard_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/writer.cpp",
        shard_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    auto compute_kernel_configuration =
        ttnn::init_device_compute_kernel_config(device->arch(), std::nullopt, MathFidelity::HiFi4, true, false, false);
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_configuration);

    [[maybe_unused]] auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/compute.cpp",
        shard_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = true,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_ct_args,
        });

    // =========================================================================
    // Set runtime args
    // =========================================================================
    // Semaphores for R1 and R2 neighbor data arrival
    auto r1_recv_sem = semaphores[0];
    auto r2_recv_sem = semaphores[1];

    // Split shard cores per link (cores_per_link already defined in relay setup)
    std::vector<CoreCoord> cores_link_1(shard_cores.begin(), shard_cores.begin() + cores_per_link);
    std::vector<CoreCoord> cores_link_2(shard_cores.begin() + cores_per_link, shard_cores.end());

    // Termination masters (first core per link)
    // We need termination masters for both FWD and BWD mux, but we'll assign
    // them to R1 and R2 based on device position
    std::vector<CoreCoord> r1_term_masters = {cores_link_1[0], cores_link_2[0]};
    std::vector<CoreCoord> r2_term_masters = {cores_link_1[0], cores_link_2[0]};

    std::vector<uint32_t> r1_term_sems;
    std::vector<uint32_t> r2_term_sems;
    for (auto& tm : r1_term_masters) {
        r1_term_sems.push_back(CreateSemaphore(program, {tm}, 0));
    }
    for (auto& tm : r2_term_masters) {
        r2_term_sems.push_back(CreateSemaphore(program, {tm}, 0));
    }

    // Set mux runtime args
    uint32_t mux_core_offset = 0;
    const auto src_node_id = mesh_device->get_fabric_node_id(device_coordinate);
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        for (uint32_t dir = 0; dir < 2; dir++) {
            CoreCoord mux_logical_core = mux_cores[mux_core_offset++];
            std::vector<uint32_t> mux_rt_args = {};
            if (dir) {  // forward
                if (forward_coord.has_value()) {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link_idx, program, {mux_logical_core});
                    tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
                }
            } else {
                if (backward_coord.has_value()) {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link_idx, program, {mux_logical_core});
                    tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
                }
            }
        }
    }

    // Set kernel runtime args per core
    std::vector<CoreCoord> all_cores;
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        const auto& cores_for_link = (link_idx == 0) ? cores_link_1 : cores_link_2;
        uint32_t worker_id = 0;

        // Get relay master info for this link
        CoreCoord relay_master_logical = relay_master_cores[link_idx];
        CoreCoord relay_master_noc = mesh_device->worker_core_from_logical_core(relay_master_logical);

        // Get relay buffer addresses for this link
        const auto& relay_buffer_addrs = relay_buffer_addrs_per_link[link_idx];

        for (auto& core : cores_for_link) {
            auto core_noc = mesh_device->worker_core_from_logical_core(core);
            bool is_relay_master = (core == relay_master_logical);
            uint32_t worker_idx_in_link = worker_id;  // 0 = relay master, 1-3 = relay workers

            // Reader runtime args
            // Include buffer addresses so reader can memcpy S and M from packet buffer
            std::vector<uint32_t> reader_rt_args = {
                r1_recv_sem.address(),               // 0: R1 neighbor semaphore
                r2_recv_sem.address(),               // 1: R2 neighbor semaphore
                r1_recv_tensor.buffer()->address(),  // 2: R1 receive buffer base address
                r2_recv_tensor.buffer()->address(),  // 3: R2 receive buffer base address
            };
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_rt_args);

            // Writer runtime args
            // Input tensor addresses for R1 send (writer reads directly, no CB sync)
            // This avoids CB contention with compute kernel which also reads input via CBs
            std::vector<uint32_t> writer_rt_args = {
                input_tensor_l.buffer()->address(),  // 0: Local L source address
                input_tensor_s.buffer()->address(),  // 1: Local S source address
                input_tensor_m.buffer()->address(),  // 2: Local M source address
                r1_recv_tensor.buffer()->address(),  // 3: R1 neighbor destination
                r1_recv_sem.address(),               // 4: R1 neighbor semaphore
                r2_recv_tensor.buffer()->address(),  // 5: R2 neighbor destination
                r2_recv_sem.address(),               // 6: R2 neighbor semaphore
                core_noc.x,                          // 7: current_core_x
                core_noc.y,                          // 8: current_core_y
                is_relay_master ? 1u : 0u,           // 9: is_relay_master flag
            };

            if (is_relay_master) {
                // Relay master runtime args:
                // - Relay buffer addresses (3 buffers for 3 relay workers)
                // - Relay semaphore addresses (3 semaphores, reused for R1 and R2)
                writer_rt_args.push_back(relay_buffer_addrs[0]);             // 10: relay_buffer_0_addr
                writer_rt_args.push_back(relay_buffer_addrs[1]);             // 11: relay_buffer_1_addr
                writer_rt_args.push_back(relay_buffer_addrs[2]);             // 12: relay_buffer_2_addr
                writer_rt_args.push_back(relay_sems_per_link[link_idx][0]);  // 13: relay_sem_0
                writer_rt_args.push_back(relay_sems_per_link[link_idx][1]);  // 14: relay_sem_1
                writer_rt_args.push_back(relay_sems_per_link[link_idx][2]);  // 15: relay_sem_2
            } else {
                // Relay worker runtime args:
                // - Relay master NOC coordinates
                // - This worker's assigned relay buffer address on master
                // - This worker's assigned relay semaphore address on master
                uint32_t relay_worker_idx = worker_idx_in_link - 1;  // 0, 1, or 2
                uint32_t my_relay_buffer_addr = relay_buffer_addrs[relay_worker_idx];
                uint32_t my_relay_sem_addr = relay_sems_per_link[link_idx][relay_worker_idx];

                writer_rt_args.push_back(relay_master_noc.x);    // 10: relay_master_noc_x
                writer_rt_args.push_back(relay_master_noc.y);    // 11: relay_master_noc_y
                writer_rt_args.push_back(my_relay_buffer_addr);  // 12: my_relay_buffer_addr
                writer_rt_args.push_back(my_relay_sem_addr);     // 13: my_relay_sem_addr
            }

            // Determine which physical mux (FWD or BWD) is used for R1 and R2
            // mux_cores layout: [link0_bwd, link0_fwd, link1_bwd, link1_fwd]
            CoreCoord fwd_mux_logical = mux_cores[(link_idx * 2) + 1];
            CoreCoord bwd_mux_logical = mux_cores[link_idx * 2];

            // Map R1/R2 to FWD/BWD based on device position
            CoreCoord r1_mux_logical = r1_uses_fwd_mux ? fwd_mux_logical : bwd_mux_logical;
            CoreCoord r2_mux_logical = r1_uses_fwd_mux ? bwd_mux_logical : fwd_mux_logical;

            CoreCoord r1_mux_virtual = mesh_device->worker_core_from_logical_core(r1_mux_logical);
            CoreCoord r2_mux_virtual = mesh_device->worker_core_from_logical_core(r2_mux_logical);

            CoreCoord r1_term_master = r1_term_masters[link_idx];
            CoreCoord r2_term_master = r2_term_masters[link_idx];
            CoreCoord r1_term_master_virtual = mesh_device->worker_core_from_logical_core(r1_term_master);
            CoreCoord r2_term_master_virtual = mesh_device->worker_core_from_logical_core(r2_term_master);

            // Add R1 mux runtime args
            // With relay optimization, only relay master uses the mux (channel 0)
            // All cores get mux args but only relay master uses them
            // Always use worker_id 0 since mux is configured for 1 client only
            constexpr uint32_t mux_worker_id = 0;
            simplified_fabric_mux_rt_args(
                is_relay_master,  // Only relay master is termination master
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                r1_mux_virtual,
                mux_worker_id,
                core,
                mux_kernel_config,
                program,
                r1_term_master_virtual,
                r1_term_sems[link_idx],
                writer_rt_args);

            // Add R2 mux runtime args
            simplified_fabric_mux_rt_args(
                is_relay_master,  // Only relay master is termination master
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                r2_mux_virtual,
                mux_worker_id,
                core,
                mux_kernel_config,
                program,
                r2_term_master_virtual,
                r2_term_sems[link_idx],
                writer_rt_args);

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core, writer_rt_args);

            all_cores.push_back(core);
            worker_id++;
        }
    }

    // =========================================================================
    // Return cached program
    // =========================================================================
    return {
        std::move(program),
        ReduceToAllOp::ReduceToAll::shared_variables_t{
            .reader_kernel1 = reader_kernel,
            .reader_kernel2 = 0,  // Not used in simplified design
            .cores1 = all_cores,
            .cores2 = {},
            .writer_kernel1 = writer_kernel,
            .writer_kernel2 = 0,  // Not used in simplified design
            .semaphores = semaphores,
            .is_device_0_2 = is_even_device,
            .is_simplified = true}};
}

}  // namespace ttnn::operations::ccl
