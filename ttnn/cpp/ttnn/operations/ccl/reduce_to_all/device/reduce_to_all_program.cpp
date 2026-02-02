// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

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

ttnn::device_operation::CachedProgram<ReduceToAllOp::ReduceToAll::shared_variables_t> reduce_to_all_program_factory(
    const ReduceToAllOp::tensor_args_t& tensor_args,
    const ReduceToAllOp::operation_attributes_t& operation_attributes,
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

    TT_FATAL(forward_coord.has_value(), "Forward coordinate must be provided for ReduceToAll operation");
    TT_FATAL(backward_coord.has_value(), "Backward coordinate must be provided for ReduceToAll operation");

    const auto forward_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
    const auto backward_node_id = mesh_device->get_fabric_node_id(backward_coord.value());

    const uint32_t device_index = device_coordinate[0];  // Assuming 1D ring on first dimension

    const auto& input_tensor_l = tensor_args.input_tensor_l;
    const auto& input_tensor_ms = tensor_args.input_tensor_ms;

    const auto& output_tensor_l = output_tensors.at(1)[0];  // Only L output (normalized)

    // Use intermediate tensors as receive buffers.
    // These are MeshDevice tensors created ONCE at the mesh level, so they have
    // the SAME L1 address on ALL devices. This is critical for fabric sends!
    //
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
    const uint32_t ms_tile_size_bytes = aligned_page_size;                       // Single combined MS tile
    const uint32_t total_packet_size = payload_size_bytes + ms_tile_size_bytes;  // L + MS

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
    constexpr auto cb_local_ms = tt::CBIndex::c_1;  // Combined max/sum

    // R1 neighbor input (aliased to R1 MeshBuffer)
    constexpr auto cb_r1_neighbor_l = tt::CBIndex::c_2;
    constexpr auto cb_r1_neighbor_ms = tt::CBIndex::c_3;  // Combined max/sum

    // R1 result / R2 local input (writer sends to R2 neighbor)
    constexpr auto cb_r1_result_l = tt::CBIndex::c_4;
    constexpr auto cb_r1_result_ms = tt::CBIndex::c_5;  // Combined max/sum

    // R2 neighbor input (aliased to R2 MeshBuffer)
    constexpr auto cb_r2_neighbor_l = tt::CBIndex::c_6;
    constexpr auto cb_r2_neighbor_ms = tt::CBIndex::c_7;  // Combined max/sum

    // Output CBs
    constexpr auto cb_l_out = tt::CBIndex::c_8;   // ALIASED to output_l
    constexpr auto cb_ms_out = tt::CBIndex::c_9;  // Intermediate MS (R1 output, not aliased)

    // Packet slot CB for writer (unified header + payload)
    constexpr auto cb_packet_slot = tt::CBIndex::c_10;

    // Sync CBs for coordination between Compute and Writer
    // cb_sync: Compute -> Writer (R1 results ready)
    // cb_sync_writer_done: Writer -> Compute (Writer finished reading R1 results)
    constexpr auto cb_sync = tt::CBIndex::c_11;
    constexpr auto cb_sync_writer_done = tt::CBIndex::c_12;

    // =========================================================================
    // Create Circular Buffers
    // =========================================================================
    // Local input CBs - ALIASED to input tensor shards
    tt::tt_metal::CircularBufferConfig cb_local_l_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_local_l, input_dataformat}})
            .set_page_size(cb_local_l, aligned_page_size)
            .set_tile_dims(cb_local_l, stats_tile)
            .set_globally_allocated_address(*input_tensor_l.buffer());
    auto cb_local_l_handle = CreateCircularBuffer(program, shard_grid, cb_local_l_config);

    tt::tt_metal::CircularBufferConfig cb_local_ms_config =
        tt::tt_metal::CircularBufferConfig(aligned_page_size, {{cb_local_ms, input_dataformat}})
            .set_page_size(cb_local_ms, aligned_page_size)
            .set_tile_dims(cb_local_ms, stats_tile)
            .set_globally_allocated_address(*input_tensor_ms.buffer());
    auto cb_local_ms_handle = CreateCircularBuffer(program, shard_grid, cb_local_ms_config);

    // R1 neighbor CBs - ALIASED to R1 receive tensor
    tt::tt_metal::CircularBufferConfig cb_r1_neighbor_l_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_r1_neighbor_l, input_dataformat}})
            .set_page_size(cb_r1_neighbor_l, aligned_page_size)
            .set_tile_dims(cb_r1_neighbor_l, stats_tile)
            .set_globally_allocated_address(*r1_recv_tensor.buffer());
    auto cb_r1_neighbor_l_handle = CreateCircularBuffer(program, shard_grid, cb_r1_neighbor_l_config);

    // MS CB is NOT aliased - reader will memcpy from buffer after packet arrives
    // (CB aliasing doesn't support offsets, and MS is tiny - ~512 bytes)
    tt::tt_metal::CircularBufferConfig cb_r1_neighbor_ms_config =
        tt::tt_metal::CircularBufferConfig(aligned_page_size, {{cb_r1_neighbor_ms, input_dataformat}})
            .set_page_size(cb_r1_neighbor_ms, aligned_page_size)
            .set_tile_dims(cb_r1_neighbor_ms, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r1_neighbor_ms_config);

    // R1 result CBs (intermediate, not aliased)
    tt::tt_metal::CircularBufferConfig cb_r1_result_l_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_r1_result_l, input_dataformat}})
            .set_page_size(cb_r1_result_l, aligned_page_size)
            .set_tile_dims(cb_r1_result_l, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r1_result_l_config);

    tt::tt_metal::CircularBufferConfig cb_r1_result_ms_config =
        tt::tt_metal::CircularBufferConfig(aligned_page_size, {{cb_r1_result_ms, input_dataformat}})
            .set_page_size(cb_r1_result_ms, aligned_page_size)
            .set_tile_dims(cb_r1_result_ms, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r1_result_ms_config);

    // R2 neighbor CBs - ALIASED to R2 receive tensor (direct fabric write, zero-copy receive)
    tt::tt_metal::CircularBufferConfig cb_r2_neighbor_l_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_r2_neighbor_l, input_dataformat}})
            .set_page_size(cb_r2_neighbor_l, aligned_page_size)
            .set_tile_dims(cb_r2_neighbor_l, stats_tile)
            .set_globally_allocated_address(*r2_recv_tensor.buffer());
    auto cb_r2_neighbor_l_handle = CreateCircularBuffer(program, shard_grid, cb_r2_neighbor_l_config);

    tt::tt_metal::CircularBufferConfig cb_r2_neighbor_ms_config =
        tt::tt_metal::CircularBufferConfig(aligned_page_size, {{cb_r2_neighbor_ms, input_dataformat}})
            .set_page_size(cb_r2_neighbor_ms, aligned_page_size)
            .set_tile_dims(cb_r2_neighbor_ms, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_r2_neighbor_ms_config);

    // Output CBs
    // cb_l_out is ALIASED to output_tensor_l shard - compute writes directly to output!
    tt::tt_metal::CircularBufferConfig cb_l_out_config =
        tt::tt_metal::CircularBufferConfig(out_tiles * aligned_page_size, {{cb_l_out, input_dataformat}})
            .set_page_size(cb_l_out, aligned_page_size)
            .set_tile_dims(cb_l_out, stats_tile)
            .set_globally_allocated_address(*output_tensor_l.buffer());
    auto cb_l_out_handle = CreateCircularBuffer(program, shard_grid, cb_l_out_config);

    // cb_ms_out is intermediate MS output from R1 (not aliased to final output)
    tt::tt_metal::CircularBufferConfig cb_ms_out_config =
        tt::tt_metal::CircularBufferConfig(aligned_page_size, {{cb_ms_out, input_dataformat}})
            .set_page_size(cb_ms_out, aligned_page_size)
            .set_tile_dims(cb_ms_out, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_ms_out_config);

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

    // Sync CBs (tiny, 1 tile size each)
    // cb_sync: Compute signals Writer that R1 results are ready
    tt::tt_metal::CircularBufferConfig cb_sync_config =
        tt::tt_metal::CircularBufferConfig(aligned_page_size, {{cb_sync, tt::DataFormat::RawUInt32}})
            .set_page_size(cb_sync, aligned_page_size)
            .set_tile_dims(cb_sync, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_sync_config);

    // cb_sync_writer_done: Writer signals Compute that it finished reading R1 results
    // This prevents race where Compute pops R1 results while Writer still reading
    tt::tt_metal::CircularBufferConfig cb_sync_writer_done_config =
        tt::tt_metal::CircularBufferConfig(aligned_page_size, {{cb_sync_writer_done, tt::DataFormat::RawUInt32}})
            .set_page_size(cb_sync_writer_done, aligned_page_size)
            .set_tile_dims(cb_sync_writer_done, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_sync_writer_done_config);

    // =========================================================================
    // Setup forwarder cores and config
    // =========================================================================
    // forwarder replaces mux with lightweight packet forwarding.
    // - 2 forwarder cores (1 per link)
    // - Each forwarder core runs BRISC (FWD) and NCRISC (BWD)
    // - Workers write complete packets to forwarder slots, forwarder forwards via fabric
    constexpr auto num_links = 2;
    const uint32_t num_workers_per_link = num_shard_cores / num_links;

    // Workers are split into Type A and Type B based on (core_index % 2).
    // Half the workers per link are Type A, half are Type B.
    // packets_per_round = workers of same type per link = num_workers_per_link / 2
    const uint32_t packets_per_round = num_workers_per_link / 2;

    // Use first 2 cores from input_forwarder_cores (renamed to forwarder cores) or default
    std::vector<CoreCoord> forwarder_cores = {CoreCoord(2, 0), CoreCoord(2, 1)};
    if (operation_attributes.input_forwarder_cores.has_value() &&
        operation_attributes.input_forwarder_cores.value().size() == 2) {
        forwarder_cores = {
            operation_attributes.input_forwarder_cores.value()[0],
            operation_attributes.input_forwarder_cores.value()[1]};
    }
    CoreRangeSet forwarder_core_range_set = CoreRangeSet(forwarder_cores);

    // forwarder buffer layout per core (68KB total, shared by BRISC and NCRISC):
    // - BRISC region (offset 0): [FWD R1 slots][FWD R2 slots] - 4 slots total
    // - NCRISC region (offset 4*slot_size): [BWD R1 slots][BWD R2 slots] - 4 slots total
    // Each slot = packet header + L + S + M (slot_size already computed and L1-aligned above)
    const uint32_t slots_per_direction = 2 * packets_per_round;  // R1 + R2 slots
    const uint32_t brisc_buffer_size = slots_per_direction * slot_size;
    const uint32_t ncrisc_buffer_offset = brisc_buffer_size;  // NCRISC starts after BRISC region

    // Allocate forwarder buffer in L1 on forwarder cores
    // Use scratch tensor if provided (preferred - decouples from semaphore allocations)
    // Otherwise fall back to l1_unreserved_base_address (legacy behavior)
    uint32_t forwarder_buffer_base = 0;
    if (tensor_args.optional_forwarder_scratch_tensor.has_value()) {
        const auto& scratch_tensor = tensor_args.optional_forwarder_scratch_tensor.value();
        forwarder_buffer_base = scratch_tensor.buffer()->address();
    } else {
        const uint32_t l1_unreserved_base_address =
            mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        forwarder_buffer_base = l1_unreserved_base_address;
    }

    // Create semaphores for forwarder synchronization (2 per forwarder core)
    // Non-blocking design: Each semaphore is bit-packed (1 << slot_idx) per worker.
    // forwarder polls mask, forwards ready slots immediately without blocking per-round.
    // Layout: [link0_fwd, link0_bwd, link1_fwd, link1_bwd]
    // Maximum 32 slots per direction (limited by 32-bit semaphore width).
    std::vector<uint32_t> forwarder_sem_addrs;
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        CoreCoord agg_core = forwarder_cores[link_idx];
        forwarder_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // fwd
        forwarder_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // bwd
    }

    // forwarder compile-time args (non-blocking design)
    // forwarder polls bit-packed semaphore and forwards any ready slots immediately.
    std::vector<uint32_t> forwarder_ct_args = {
        slots_per_direction,  // 0: Total slots per direction (R1 + R2 combined)
        slot_size,            // 1: Total packet size (header + L + MS)
    };

    // =========================================================================
    // Create kernel compile-time args
    // =========================================================================

    // Reader compile-time args
    // Note: cb_r1_neighbor_l and cb_r2_neighbor_l are aliased to receive buffers (zero-copy for L)
    // MS needs memcpy from buffer since we can't alias at offsets
    std::vector<uint32_t> reader_ct_args = {
        Sq_chunk_t,             // 0
        vDHt,                   // 1
        input_page_size_bytes,  // 2
        l1_alignment,           // 3
        cb_local_l,             // 4
        cb_local_ms,            // 5
        cb_r1_neighbor_l,       // 6
        cb_r1_neighbor_ms,      // 7
        cb_r2_neighbor_l,       // 8
        cb_r2_neighbor_ms,      // 9
        payload_size_bytes,     // 10 - offset to MS data in receive buffer
        ms_tile_size_bytes,     // 11 - size of MS data to copy
    };

    // Writer compile-time args
    // The writer builds packets and writes them to forwarder slots via NoC,
    // then increments forwarder semaphore.
    std::vector<uint32_t> writer_ct_args = {
        Sq_chunk_t,             // 0
        vDHt,                   // 1
        cb_local_l,             // 2
        cb_local_ms,            // 3
        cb_r1_result_l,         // 4
        cb_r1_result_ms,        // 5
        cb_packet_slot,         // 6: Unified packet slot CB (header + payload)
        l1_alignment,           // 7
        input_page_size_bytes,  // 8
        cb_sync,                // 9: Compute -> Writer (R1 ready)
        slot_size,              // 10: forwarder slot size (L1-aligned)
        cb_sync_writer_done,    // 11: Writer -> Compute (done reading R1 results)
    };

    // Compute compile-time args (15 total for optimized compute)
    // Layout matches compute.cpp expectations:
    // - 0-1: Local input CBs (L, MS)
    // - 2-3: R1 neighbor CBs
    // - 4-5: R1 result CBs
    // - 6-7: R2 neighbor CBs
    // - 8-9: Output CBs (L, MS intermediate)
    // - 10-14: scale, block_size, num_blocks, cb_sync, cb_sync_writer_done
    std::vector<uint32_t> compute_ct_args = {
        cb_local_l,           // 0
        cb_local_ms,          // 1
        cb_r1_neighbor_l,     // 2
        cb_r1_neighbor_ms,    // 3
        cb_r1_result_l,       // 4
        cb_r1_result_ms,      // 5
        cb_r2_neighbor_l,     // 6
        cb_r2_neighbor_ms,    // 7
        cb_l_out,             // 8 - ALIASED to output_l
        cb_ms_out,            // 9 - intermediate MS
        scale_val,            // 10
        vDHt,                 // 11 - block_size (tiles per row)
        Sq_chunk_t,           // 12 - num_blocks (number of rows)
        cb_sync,              // 13 - Compute -> Writer (R1 ready)
        cb_sync_writer_done,  // 14 - Writer -> Compute (done reading R1 results)
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

    [[maybe_unused]] auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/compute.cpp",
        shard_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
        });

    // Create forwarder kernels (BRISC for FWD, NCRISC for BWD)
    // Both use the same kernel code - direction is implicit in fabric connection rt args
    auto forwarder_brisc_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/forwarder.cpp",
        forwarder_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = forwarder_ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    auto forwarder_ncrisc_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_all/device/kernels/forwarder.cpp",
        forwarder_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = forwarder_ct_args,
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

    // Set forwarder runtime args (BRISC for FWD, NCRISC for BWD)
    // Non-blocking design: single semaphore per direction (bit-packed by workers)
    const auto src_node_id = mesh_device->get_fabric_node_id(device_coordinate);
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        CoreCoord agg_core = forwarder_cores[link_idx];

        // Semaphore addresses for this forwarder core (2 per core: fwd, bwd)
        uint32_t fwd_sem = forwarder_sem_addrs[link_idx * 2 + 0];
        uint32_t bwd_sem = forwarder_sem_addrs[link_idx * 2 + 1];

        // BRISC runtime args (FWD direction)
        // buffer_offset = 0 (BRISC uses first region)
        std::vector<uint32_t> brisc_rt_args = {
            forwarder_buffer_base,  // 0: buffer_base
            0,                      // 1: buffer_offset (BRISC uses offset 0)
            fwd_sem,                // 2: semaphore (bit-packed by workers)
        };
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_node_id, forward_node_id, link_idx, program, agg_core, brisc_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, forwarder_brisc_kernel, agg_core, brisc_rt_args);

        // NCRISC runtime args (BWD direction)
        // buffer_offset = ncrisc_buffer_offset (NCRISC uses second region)
        std::vector<uint32_t> ncrisc_rt_args = {
            forwarder_buffer_base,  // 0: buffer_base
            ncrisc_buffer_offset,   // 1: buffer_offset (NCRISC uses offset after BRISC region)
            bwd_sem,                // 2: semaphore (bit-packed by workers)
        };
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_node_id, backward_node_id, link_idx, program, agg_core, ncrisc_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, forwarder_ncrisc_kernel, agg_core, ncrisc_rt_args);
    }

    // Set kernel runtime args per worker core
    std::vector<CoreCoord> all_cores;
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        const auto& cores_for_link = (link_idx == 0) ? cores_link_1 : cores_link_2;
        CoreCoord agg_core = forwarder_cores[link_idx];
        CoreCoord agg_core_noc = mesh_device->worker_core_from_logical_core(agg_core);

        // Semaphore addresses for this forwarder core (2 per core: fwd, bwd)
        uint32_t fwd_sem = forwarder_sem_addrs[link_idx * 2 + 0];
        uint32_t bwd_sem = forwarder_sem_addrs[link_idx * 2 + 1];

        // Track slot indices for R1 and R2 within each direction
        // Each direction has slots_per_direction slots: [R1 slots: 0..ppr-1][R2 slots: ppr..2*ppr-1]
        // where ppr = packets_per_round
        uint32_t fwd_r1_slot_idx = 0;  // FWD direction R1 slots
        uint32_t fwd_r2_slot_idx = 0;  // FWD direction R2 slots
        uint32_t bwd_r1_slot_idx = 0;  // BWD direction R1 slots
        uint32_t bwd_r2_slot_idx = 0;  // BWD direction R2 slots

        for (uint32_t worker_idx = 0; worker_idx < cores_for_link.size(); worker_idx++) {
            CoreCoord core = cores_for_link[worker_idx];
            auto core_noc = mesh_device->worker_core_from_logical_core(core);

            // Determine worker type: Type A if (device_id + core_index) % 2 == 0
            // Type A: R1=FWD, R2=BWD; Type B: R1=BWD, R2=FWD
            const bool is_type_a = ((device_index + worker_idx) % 2) == 0;

            // Assign slot indices within each direction's slot space
            // R1 slots: 0..packets_per_round-1, R2 slots: packets_per_round..2*packets_per_round-1
            uint32_t r1_slot_idx;
            uint32_t r2_slot_idx;
            uint32_t r1_agg_sem;
            uint32_t r2_agg_sem;
            auto r1_dst_node_id = forward_node_id;   // Default to forward
            auto r2_dst_node_id = backward_node_id;  // Default to backward

            if (is_type_a) {
                // Type A: R1=FWD, R2=BWD
                r1_slot_idx = fwd_r1_slot_idx++;
                r2_slot_idx = packets_per_round + bwd_r2_slot_idx++;
                r1_agg_sem = fwd_sem;
                r2_agg_sem = bwd_sem;
                r1_dst_node_id = forward_node_id;
                r2_dst_node_id = backward_node_id;
            } else {
                // Type B: R1=BWD, R2=FWD
                r1_slot_idx = bwd_r1_slot_idx++;
                r2_slot_idx = packets_per_round + fwd_r2_slot_idx++;
                r1_agg_sem = bwd_sem;
                r2_agg_sem = fwd_sem;
                r1_dst_node_id = backward_node_id;
                r2_dst_node_id = forward_node_id;
            }

            // Calculate forwarder slot addresses based on direction
            uint32_t r1_slot_base = is_type_a ? forwarder_buffer_base : (forwarder_buffer_base + ncrisc_buffer_offset);
            uint32_t r2_slot_base = is_type_a ? (forwarder_buffer_base + ncrisc_buffer_offset) : forwarder_buffer_base;

            // Calculate actual slot addresses (R1 slots first, then R2 slots in each region)
            // Type A R1: FWD region slot r1_slot_idx
            // Type A R2: BWD region slot (r2_slot_idx - packets_per_round) + packets_per_round offset
            uint32_t r1_slot_addr = r1_slot_base + (r1_slot_idx * slot_size);
            uint32_t r2_slot_addr = r2_slot_base + (r2_slot_idx * slot_size);

            // Reader runtime args
            std::vector<uint32_t> reader_rt_args = {
                r1_recv_sem.address(),               // 0: R1 neighbor semaphore
                r2_recv_sem.address(),               // 1: R2 neighbor semaphore
                r1_recv_tensor.buffer()->address(),  // 2: R1 receive buffer base address
                r2_recv_tensor.buffer()->address(),  // 3: R2 receive buffer base address
            };
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_rt_args);

            // Writer runtime args (with slot indices for bit-packed semaphore signaling)
            std::vector<uint32_t> writer_rt_args = {
                input_tensor_l.buffer()->address(),   // 0: Local L source address
                input_tensor_ms.buffer()->address(),  // 1: Local MS source address
                *r1_dst_node_id.mesh_id,              // 2: R1 neighbor destination mesh ID
                r1_dst_node_id.chip_id,               // 3: R1 neighbor destination chip ID
                r1_recv_tensor.buffer()->address(),   // 4: R1 neighbor destination
                r1_recv_sem.address(),                // 5: R1 neighbor semaphore
                *r2_dst_node_id.mesh_id,              // 6: R2 neighbor destination mesh ID
                r2_dst_node_id.chip_id,               // 7: R2 neighbor destination chip ID
                r2_recv_tensor.buffer()->address(),   // 8: R2 neighbor destination
                r2_recv_sem.address(),                // 9: R2 neighbor semaphore
                core_noc.x,                           // 10: current_core_x
                core_noc.y,                           // 11: current_core_y
                // forwarder-specific args
                agg_core_noc.x,  // 12: forwarder_core_x
                agg_core_noc.y,  // 13: forwarder_core_y
                r1_slot_addr,    // 14: R1 forwarder slot address
                r1_agg_sem,      // 15: R1 forwarder semaphore
                r1_slot_idx,     // 16: R1 slot index (for bit-packed signaling: 1 << r1_slot_idx)
                r2_slot_addr,    // 17: R2 forwarder slot address
                r2_agg_sem,      // 18: R2 forwarder semaphore
                r2_slot_idx,     // 19: R2 slot index (for bit-packed signaling: 1 << r2_slot_idx)
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
            .reader_kernel = reader_kernel,
            .worker_cores = all_cores,
            .writer_kernel = writer_kernel,
            .semaphores = semaphores,
            // CB handles for trace replay (UpdateDynamicCircularBufferAddressAndTotalSize)
            .cb_local_l_handle = cb_local_l_handle,
            .cb_local_ms_handle = cb_local_ms_handle,
            .cb_r1_neighbor_l_handle = cb_r1_neighbor_l_handle,
            .cb_r2_neighbor_l_handle = cb_r2_neighbor_l_handle,
            .cb_l_out_handle = cb_l_out_handle,
            .l_tile_size = out_tiles * aligned_page_size,
            .ms_tile_size = aligned_page_size,
            .out_tiles = out_tiles}};
}

void ReduceToAllOp::ReduceToAll::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /* operation_attributes */,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        // Get the input tensors
        const auto& input_tensor_l = tensor_args.input_tensor_l;
        const auto& input_tensor_ms = tensor_args.input_tensor_ms;

        // Get output tensors
        const auto& output_tensors_l = tensor_return_value[1];
        const auto& intermediate_tensors = tensor_return_value[0];

        for (const auto& core : shared_variables.worker_cores) {
            // reader runtime args:
            // 0: R1 neighbor semaphore, 1: R2 neighbor semaphore
            // 2: R1 receive buffer, 3: R2 receive buffer
            auto& reader_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_variables.reader_kernel);
            auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];
            reader_runtime_args[0] = shared_variables.semaphores[0].address();     // R1 recv sem
            reader_runtime_args[1] = shared_variables.semaphores[1].address();     // R2 recv sem
            reader_runtime_args[2] = intermediate_tensors[0].buffer()->address();  // R1 recv buffer
            reader_runtime_args[3] = intermediate_tensors[1].buffer()->address();  // R2 recv buffer

            // writer runtime args layout:
            // 0: input L, 1: input MS
            // 2: R1 mesh_id (static), 3: R1 chip_id (static)
            // 4: R1 dest addr, 5: R1 sem addr
            // 6: R2 mesh_id (static), 7: R2 chip_id (static)
            // 8: R2 dest addr, 9: R2 sem addr
            // 10-19: forwarder args (static)
            auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_variables.writer_kernel);
            auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
            writer_runtime_args[0] = input_tensor_l.buffer()->address();
            writer_runtime_args[1] = input_tensor_ms.buffer()->address();
            // Indices 2-3 (mesh_id, chip_id) are static - don't update
            writer_runtime_args[4] = intermediate_tensors[0].buffer()->address();  // R1 dest
            writer_runtime_args[5] = shared_variables.semaphores[0].address();     // R1 sem
            // Indices 6-7 (mesh_id, chip_id) are static - don't update
            writer_runtime_args[8] = intermediate_tensors[1].buffer()->address();  // R2 dest
            writer_runtime_args[9] = shared_variables.semaphores[1].address();     // R2 sem
        }

        // Update CB addresses for aliased buffers (critical for trace replay)
        // When trace replays, the CBs still point to old buffer addresses unless updated
        if (shared_variables.cb_local_l_handle.has_value()) {
            UpdateDynamicCircularBufferAddressAndTotalSize(
                program,
                shared_variables.cb_local_l_handle.value(),
                *input_tensor_l.buffer(),
                shared_variables.l_tile_size);
        }
        if (shared_variables.cb_local_ms_handle.has_value()) {
            UpdateDynamicCircularBufferAddressAndTotalSize(
                program,
                shared_variables.cb_local_ms_handle.value(),
                *input_tensor_ms.buffer(),
                shared_variables.ms_tile_size);
        }
        if (shared_variables.cb_r1_neighbor_l_handle.has_value()) {
            UpdateDynamicCircularBufferAddressAndTotalSize(
                program,
                shared_variables.cb_r1_neighbor_l_handle.value(),
                *intermediate_tensors[0].buffer(),
                shared_variables.l_tile_size);
        }
        if (shared_variables.cb_r2_neighbor_l_handle.has_value()) {
            UpdateDynamicCircularBufferAddressAndTotalSize(
                program,
                shared_variables.cb_r2_neighbor_l_handle.value(),
                *intermediate_tensors[1].buffer(),
                shared_variables.l_tile_size);
        }
        if (shared_variables.cb_l_out_handle.has_value()) {
            UpdateDynamicCircularBufferAddressAndTotalSize(
                program,
                shared_variables.cb_l_out_handle.value(),
                *output_tensors_l[0].buffer(),
                shared_variables.l_tile_size);
        }
    }
};

}  // namespace ttnn::operations::ccl
