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

// NOTE: Mux helper functions removed - replaced with aggregator-based approach
// The aggregator is much simpler: workers write packets to aggregator L1 slots
// via NoC and increment semaphore, aggregator forwards packets via fabric.

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

    // Packet slot CB for writer (unified header + payload)
    constexpr auto cb_packet_slot = tt::CBIndex::c_20;

    // Sync CB for coordination between Compute and Writer
    constexpr auto cb_sync = tt::CBIndex::c_21;

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

    // Packet slot CB for writer (unified header + payload in single buffer)
    // This enables single NOC transfer instead of separate header + payload writes
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    const uint32_t slot_size_unaligned = packet_header_size_bytes + total_packet_size;
    const uint32_t slot_size = tt::round_up(slot_size_unaligned, l1_alignment);
    TT_FATAL(
        slot_size == slot_size_unaligned,
        "Slot size ({}) must be L1 aligned ({}). Header={}, payload={}",
        slot_size_unaligned,
        l1_alignment,
        packet_header_size_bytes,
        total_packet_size);

    tt::tt_metal::CircularBufferConfig cb_packet_slot_config =
        tt::tt_metal::CircularBufferConfig(2 * slot_size, {{cb_packet_slot, tt::DataFormat::RawUInt32}})
            .set_page_size(cb_packet_slot, slot_size)
            .set_tile_dims(cb_packet_slot, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_packet_slot_config);

    // Sync CB (tiny, 1 tile size, can use 16B if supported but using aligned_page_size for safety)
    tt::tt_metal::CircularBufferConfig cb_sync_config =
        tt::tt_metal::CircularBufferConfig(aligned_page_size, {{cb_sync, tt::DataFormat::RawUInt32}})
            .set_page_size(cb_sync, aligned_page_size)
            .set_tile_dims(cb_sync, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_sync_config);

    // =========================================================================
    // Setup aggregator cores and config
    // =========================================================================
    // Aggregator replaces heavyweight mux with lightweight packet forwarding.
    // - 2 aggregator cores (1 per link) instead of 4 mux cores
    // - Each aggregator core runs BRISC (FWD) and NCRISC (BWD)
    // - Workers write complete packets to aggregator slots, aggregator forwards via fabric
    constexpr auto num_links = 2;
    const uint32_t num_workers_per_link = num_shard_cores / num_links;

    // Workers are split into Type A and Type B based on (core_index % 2).
    // Half the workers per link are Type A, half are Type B.
    // packets_per_round = workers of same type per link = num_workers_per_link / 2
    const uint32_t packets_per_round = num_workers_per_link / 2;

    // Use first 2 cores from input_mux_cores (renamed to aggregator cores) or default
    std::vector<CoreCoord> aggregator_cores = {CoreCoord(2, 0), CoreCoord(2, 1)};
    if (operation_attributes.input_mux_cores.has_value() && operation_attributes.input_mux_cores.value().size() >= 2) {
        // Use first 2 cores only (aggregator needs 2, not 4 like mux)
        aggregator_cores = {
            operation_attributes.input_mux_cores.value()[0], operation_attributes.input_mux_cores.value()[1]};
    }
    CoreRangeSet aggregator_core_range_set = CoreRangeSet(aggregator_cores);

    // Aggregator buffer layout per core (68KB total, shared by BRISC and NCRISC):
    // - BRISC region (offset 0): [FWD R1 slots][FWD R2 slots] - 4 slots total
    // - NCRISC region (offset 4*slot_size): [BWD R1 slots][BWD R2 slots] - 4 slots total
    // Each slot = packet header + L + S + M (slot_size already computed and L1-aligned above)
    const uint32_t slots_per_direction = 2 * packets_per_round;  // R1 + R2 slots
    const uint32_t brisc_buffer_size = slots_per_direction * slot_size;
    const uint32_t ncrisc_buffer_offset = brisc_buffer_size;  // NCRISC starts after BRISC region

    // Allocate aggregator buffer in L1 on aggregator cores
    // Use scratch tensor if provided (preferred - decouples from semaphore allocations)
    // Otherwise fall back to l1_unreserved_base_address (legacy behavior)
    uint32_t aggregator_buffer_base = 0;
    if (tensor_args.optional_aggregator_scratch_tensor.has_value()) {
        const auto& scratch_tensor = tensor_args.optional_aggregator_scratch_tensor.value();
        aggregator_buffer_base = scratch_tensor.buffer()->address();
    } else {
        const uint32_t l1_unreserved_base_address =
            mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        aggregator_buffer_base = l1_unreserved_base_address;
    }

    // Create semaphores for aggregator synchronization (4 per aggregator core)
    // Layout: [link0_fwd_r1, link0_fwd_r2, link0_bwd_r1, link0_bwd_r2,
    //          link1_fwd_r1, link1_fwd_r2, link1_bwd_r1, link1_bwd_r2]
    std::vector<uint32_t> aggregator_sem_addrs;
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        CoreCoord agg_core = aggregator_cores[link_idx];
        // FWD R1, FWD R2, BWD R1, BWD R2
        aggregator_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // fwd_r1
        aggregator_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // fwd_r2
        aggregator_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // bwd_r1
        aggregator_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // bwd_r2
    }

    // Aggregator compile-time args
    std::vector<uint32_t> aggregator_ct_args = {
        packets_per_round,  // 0: Workers per round (half the workers per link)
        slot_size,          // 1: Total packet size (header + L + S + M)
    };

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

    // Writer compile-time args (simplified for aggregator - no mux config needed)
    // The writer builds packets and writes them to aggregator slots via NoC,
    // then increments aggregator semaphore. Much simpler than mux client protocol.
    std::vector<uint32_t> writer_ct_args = {
        Sq_chunk_t,             // 0
        vDHt,                   // 1
        cb_local_l,             // 2
        cb_local_s,             // 3
        cb_local_m,             // 4
        cb_r1_result_l,         // 5
        cb_r1_result_s,         // 6
        cb_r1_result_m,         // 7
        cb_packet_slot,         // 8: Unified packet slot CB (header + payload)
        l1_alignment,           // 9
        input_page_size_bytes,  // 10
        cb_sync,                // 11
        slot_size,              // 12: Aggregator slot size (L1-aligned)
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
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/writer_aggregator.cpp",
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

    // Create aggregator kernels (BRISC for FWD, NCRISC for BWD)
    // Both use the same kernel code - direction is implicit in fabric connection rt args
    auto aggregator_brisc_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/aggregator.cpp",
        aggregator_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = aggregator_ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    auto aggregator_ncrisc_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/aggregator.cpp",
        aggregator_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = aggregator_ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    // =========================================================================
    // Set runtime args
    // =========================================================================
    // Semaphores for R1 and R2 neighbor data arrival
    auto r1_recv_sem = semaphores[0];
    auto r2_recv_sem = semaphores[1];

    // Split shard cores per link
    const uint32_t cores_per_link = num_shard_cores / num_links;
    std::vector<CoreCoord> cores_link_1(shard_cores.begin(), shard_cores.begin() + cores_per_link);
    std::vector<CoreCoord> cores_link_2(shard_cores.begin() + cores_per_link, shard_cores.end());

    // Set aggregator runtime args (BRISC for FWD, NCRISC for BWD)
    const auto src_node_id = mesh_device->get_fabric_node_id(device_coordinate);
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        CoreCoord agg_core = aggregator_cores[link_idx];

        // Semaphore addresses for this aggregator core
        uint32_t fwd_r1_sem = aggregator_sem_addrs[link_idx * 4 + 0];
        uint32_t fwd_r2_sem = aggregator_sem_addrs[link_idx * 4 + 1];
        uint32_t bwd_r1_sem = aggregator_sem_addrs[link_idx * 4 + 2];
        uint32_t bwd_r2_sem = aggregator_sem_addrs[link_idx * 4 + 3];

        // BRISC runtime args (FWD direction)
        // buffer_offset = 0 (BRISC uses first region)
        std::vector<uint32_t> brisc_rt_args = {
            aggregator_buffer_base,  // 0: buffer_base
            0,                       // 1: buffer_offset (BRISC uses offset 0)
            fwd_r1_sem,              // 2: R1 semaphore
            fwd_r2_sem,              // 3: R2 semaphore
        };
        // Append fabric connection args (direction implicit in src→dst)
        if (forward_coord.has_value()) {
            const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_node_id, dst_node_id, link_idx, program, agg_core, brisc_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, aggregator_brisc_kernel, agg_core, brisc_rt_args);

        // NCRISC runtime args (BWD direction)
        // buffer_offset = ncrisc_buffer_offset (NCRISC uses second region)
        std::vector<uint32_t> ncrisc_rt_args = {
            aggregator_buffer_base,  // 0: buffer_base
            ncrisc_buffer_offset,    // 1: buffer_offset (NCRISC uses offset after BRISC region)
            bwd_r1_sem,              // 2: R1 semaphore
            bwd_r2_sem,              // 3: R2 semaphore
        };
        // Append fabric connection args (direction implicit in src→dst)
        if (backward_coord.has_value()) {
            const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_node_id, dst_node_id, link_idx, program, agg_core, ncrisc_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, aggregator_ncrisc_kernel, agg_core, ncrisc_rt_args);
    }

    // Set kernel runtime args per worker core
    std::vector<CoreCoord> all_cores;
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        const auto& cores_for_link = (link_idx == 0) ? cores_link_1 : cores_link_2;
        CoreCoord agg_core = aggregator_cores[link_idx];
        CoreCoord agg_core_noc = mesh_device->worker_core_from_logical_core(agg_core);

        // Semaphore addresses for this aggregator core
        uint32_t fwd_r1_sem = aggregator_sem_addrs[link_idx * 4 + 0];
        uint32_t fwd_r2_sem = aggregator_sem_addrs[link_idx * 4 + 1];
        uint32_t bwd_r1_sem = aggregator_sem_addrs[link_idx * 4 + 2];
        uint32_t bwd_r2_sem = aggregator_sem_addrs[link_idx * 4 + 3];

        // Track slot indices for each worker type
        uint32_t type_a_slot_idx = 0;  // Type A workers use slots 0, 1, ...
        uint32_t type_b_slot_idx = 0;  // Type B workers use slots 0, 1, ...

        for (uint32_t worker_idx = 0; worker_idx < cores_for_link.size(); worker_idx++) {
            CoreCoord core = cores_for_link[worker_idx];
            auto core_noc = mesh_device->worker_core_from_logical_core(core);

            // Determine worker type: Type A if (device_id + core_index) % 2 == 0
            // Type A: R1=FWD, R2=BWD; Type B: R1=BWD, R2=FWD
            const bool is_type_a = ((device_index + worker_idx) % 2) == 0;

            // Calculate slot index within this worker's type
            uint32_t my_slot_idx = is_type_a ? type_a_slot_idx++ : type_b_slot_idx++;

            // Calculate aggregator slot addresses for R1 and R2
            // Type A: R1→FWD (BRISC region, offset 0), R2→BWD (NCRISC region, offset ncrisc_buffer_offset)
            // Type B: R1→BWD (NCRISC region), R2→FWD (BRISC region)
            uint32_t r1_slot_base;
            uint32_t r2_slot_base;
            uint32_t r1_agg_sem;
            uint32_t r2_agg_sem;

            if (is_type_a) {
                // Type A: R1=FWD, R2=BWD
                r1_slot_base = aggregator_buffer_base + 0;                     // BRISC region
                r2_slot_base = aggregator_buffer_base + ncrisc_buffer_offset;  // NCRISC region
                r1_agg_sem = fwd_r1_sem;
                r2_agg_sem = bwd_r2_sem;
            } else {
                // Type B: R1=BWD, R2=FWD
                r1_slot_base = aggregator_buffer_base + ncrisc_buffer_offset;  // NCRISC region
                r2_slot_base = aggregator_buffer_base + 0;                     // BRISC region
                r1_agg_sem = bwd_r1_sem;
                r2_agg_sem = fwd_r2_sem;
            }

            // Calculate actual slot addresses (R1 slots, then R2 slots in each region)
            uint32_t r1_slot_addr = r1_slot_base + (my_slot_idx * slot_size);
            uint32_t r2_slot_addr = r2_slot_base + (packets_per_round * slot_size) + (my_slot_idx * slot_size);

            // Reader runtime args
            std::vector<uint32_t> reader_rt_args = {
                r1_recv_sem.address(),               // 0: R1 neighbor semaphore
                r2_recv_sem.address(),               // 1: R2 neighbor semaphore
                r1_recv_tensor.buffer()->address(),  // 2: R1 receive buffer base address
                r2_recv_tensor.buffer()->address(),  // 3: R2 receive buffer base address
            };
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_rt_args);

            // Writer runtime args (simplified for aggregator - no mux protocol needed)
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
                // Aggregator-specific args
                agg_core_noc.x,  // 9: aggregator_core_x
                agg_core_noc.y,  // 10: aggregator_core_y
                r1_slot_addr,    // 11: R1 aggregator slot address
                r1_agg_sem,      // 12: R1 aggregator semaphore
                r2_slot_addr,    // 13: R2 aggregator slot address
                r2_agg_sem,      // 14: R2 aggregator semaphore
            };

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core, writer_rt_args);

            all_cores.push_back(core);
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
