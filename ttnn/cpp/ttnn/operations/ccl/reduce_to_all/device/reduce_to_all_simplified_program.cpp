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
// 5. NO MUX - relay master connects directly to fabric router
//
// ARCHITECTURE:
// - Each shard core runs: reader, compute, writer
// - Reader: signals local input ready, waits for neighbor data
// - Compute: R1 reduction, then R2 reduction, then final normalization
// - Writer: sends local data to R1 neighbor, sends R1 result to R2 neighbor
// - Relay master (1 per link) connects directly to fabric router
// - Relay workers send packets to relay master via NOC
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

// Helper to add fabric router connection runtime args for relay master
// This replaces the mux connection - relay master now connects directly to fabric router
void append_fabric_router_rt_args(
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id,
    const tt::tt_fabric::FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    tt::tt_metal::Program& program,
    const CoreCoord& worker_logical_core,
    std::vector<uint32_t>& worker_rt_args) {
    // Use the fabric API to append connection args
    tt::tt_fabric::append_fabric_connection_rt_args(
        src_fabric_node_id,
        dst_fabric_node_id,
        link_idx,
        program,
        worker_logical_core,
        worker_rt_args,
        tt::CoreType::WORKER);
}

}  // anonymous namespace

ttnn::device_operation::CachedProgram<ReduceToAllOp::ReduceToAll::shared_variables_t>
reduce_to_all_simplified_program_factory(
    const ReduceToAllOp::tensor_args_t& tensor_args,
    [[maybe_unused]] const ReduceToAllOp::operation_attributes_t& operation_attributes,
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
    // NO MUX - Relay master connects directly to fabric router
    // =========================================================================
    // Phase 2 optimization: removed mux entirely.
    // Relay master now connects directly to the fabric router using WorkerToFabricEdmSender.
    // This eliminates the mux kernel and reduces latency.
    // Note: Fabric router connection info is read from L1 static memory (MEM_TENSIX_FABRIC_CONNECTIONS_BASE)
    // by WorkerToFabricEdmSender::build_from_args<TENSIX>(), so no compile-time args needed here.

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
    // Phase 2: No mux - relay master connects directly to fabric router
    // Note: Fabric router connection info is read from L1 static memory by build_from_args<TENSIX>()
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

    // Get fabric node IDs for fabric router connections
    const auto src_node_id = mesh_device->get_fabric_node_id(device_coordinate);
    std::optional<tt::tt_fabric::FabricNodeId> r1_dst_node_id;
    std::optional<tt::tt_fabric::FabricNodeId> r2_dst_node_id;

    // Determine R1 and R2 destinations based on device position
    // Even devices: R1=FWD, R2=BWD; Odd devices: R1=BWD, R2=FWD
    if (r1_uses_fwd_mux) {
        // Even device: R1 goes forward, R2 goes backward
        if (forward_coord.has_value()) {
            r1_dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
        }
        if (backward_coord.has_value()) {
            r2_dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
        }
    } else {
        // Odd device: R1 goes backward, R2 goes forward
        if (backward_coord.has_value()) {
            r1_dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
        }
        if (forward_coord.has_value()) {
            r2_dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
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

                // Phase 2: Add fabric router connection args for R1 and R2
                // Relay master connects directly to fabric router (no mux)
                // Each connection needs: eth_channel, teardown_sem, buffer_idx_sem (appended by
                // append_fabric_router_rt_args)
                if (r1_dst_node_id.has_value()) {
                    append_fabric_router_rt_args(
                        src_node_id, r1_dst_node_id.value(), link_idx, program, core, writer_rt_args);
                }
                if (r2_dst_node_id.has_value()) {
                    append_fabric_router_rt_args(
                        src_node_id, r2_dst_node_id.value(), link_idx, program, core, writer_rt_args);
                }
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
                // Relay workers don't need fabric router args - they send to relay master
            }

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
