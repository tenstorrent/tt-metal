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
    //
    // - R1 receive buffer: Use fw_intermediate_tensor (for R1 neighbor data)
    // - R2 receive buffer: Use bw_intermediate_tensor (for R2 neighbor data)
    //
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

    // const uint32_t payload_size_bytes = out_tiles * input_page_size_bytes;
    const uint32_t ms_tile_size_bytes = aligned_page_size;                       // Single combined MS tile
    // const uint32_t total_packet_size = payload_size_bytes + ms_tile_size_bytes;  // L + MS

    // =========================================================================
    // Chunking configuration
    // =========================================================================
    // Chunking hides fabric transfer latency by overlapping data transfer with compute.
    // - MS packet sent first (slot 0)
    // - L chunks sent after (slots 1..num_l_chunks)
    // Buffer layout on receiver: [L_chunk0][L_chunk1]...[L_chunkN-1][MS]
    //
    // Block size derivation:
    // - block_size = tiles_per_l_chunk (each chunk = one SDPA block)
    // - Hardware constraint: block_size <= 8 (DST register limit)
    // - This means: num_l_chunks >= ceil(out_tiles / 8)
    //
    // For out_tiles=16: num_l_chunks >= 2 (block_size=8)
    constexpr uint32_t max_block_size = 8;  // DST register limit
    const uint32_t min_num_l_chunks = (out_tiles + max_block_size - 1) / max_block_size;

    // Configure num_l_chunks (must satisfy hardware constraint)
    // const uint32_t num_l_chunks = min_num_l_chunks;  // Use minimum for now (largest valid block_size)
    const uint32_t num_l_chunks = 4;

    TT_FATAL(num_l_chunks >= 1, "num_l_chunks must be at least 1");
    TT_FATAL(
        out_tiles % num_l_chunks == 0,
        "out_tiles ({}) must be divisible by num_l_chunks ({})",
        out_tiles,
        num_l_chunks);

    const uint32_t tiles_per_l_chunk = out_tiles / num_l_chunks;
    const uint32_t l_chunk_size_bytes = tiles_per_l_chunk * input_page_size_bytes;

    // Derive block_size from chunking (each chunk = one SDPA block)
    const uint32_t block_size = tiles_per_l_chunk;
    const uint32_t num_blocks = num_l_chunks;  // blocks_per_chunk = 1, so num_blocks = num_l_chunks

    TT_FATAL(
        block_size <= max_block_size,
        "block_size ({}) exceeds maximum ({}). Increase num_l_chunks to at least {}.",
        block_size,
        max_block_size,
        min_num_l_chunks);

    // Slot sizing: largest packet is L chunk (MS is smaller)
    // All slots sized for L chunk to simplify buffer management
    const size_t max_fabric_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const uint32_t max_chunk_payload_size = l_chunk_size_bytes;
    TT_FATAL(
        max_chunk_payload_size <= max_fabric_payload_size,
        "L chunk payload ({} bytes) exceeds fabric max ({})",
        max_chunk_payload_size,
        max_fabric_payload_size);

    log_info(
        tt::LogTest,
        "Tile width={}, height={}, ms_tile_size_bytes={}, out_tiles={}, input_page_size_bytes={}, aligned_page_size={}",
        tile_width,
        tile_height,
        ms_tile_size_bytes,
        out_tiles,
        input_page_size_bytes,
        aligned_page_size);
    log_info(
        tt::LogTest,
        "Chunking: num_l_chunks={}, tiles_per_l_chunk={}, block_size={}, num_blocks={}, l_chunk_size_bytes={}, "
        "max_chunk_payload_size={}",
        num_l_chunks,
        tiles_per_l_chunk,
        block_size,
        num_blocks,
        l_chunk_size_bytes,
        max_chunk_payload_size);

    // Scale encoding
    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = scale_fp32;
    uint32_t scale_val = scale_union.u;

    tt::tt_metal::Program program{};

    const auto tiny_tile = tt::tt_metal::Tile({8, 32});
    auto stats_tile = tiny_tile;
    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_l.dtype());

    // =========================================================================
    // CB indices
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
    // Slot size based on largest chunk (L chunk, since MS is smaller)
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    const uint32_t slot_size_unaligned = packet_header_size_bytes + max_chunk_payload_size;
    const uint32_t slot_size = tt::round_up(slot_size_unaligned, l1_alignment);
    log_info(
        tt::LogTest,
        "ReduceToAll packet slot size: header={} + max_payload={} = total={} (aligned to {})",
        packet_header_size_bytes,
        max_chunk_payload_size,
        slot_size_unaligned,
        slot_size);
    TT_FATAL(
        slot_size == slot_size_unaligned,
        "Slot size ({}) must be L1 aligned ({}). Header={}, max_payload={}",
        slot_size_unaligned,
        l1_alignment,
        packet_header_size_bytes,
        max_chunk_payload_size);

    tt::tt_metal::CircularBufferConfig cb_packet_slot_config =
        tt::tt_metal::CircularBufferConfig(2 * slot_size, {{cb_packet_slot, tt::DataFormat::RawUInt32}})
            .set_page_size(cb_packet_slot, slot_size)
            .set_tile_dims(cb_packet_slot, stats_tile);
    CreateCircularBuffer(program, shard_grid, cb_packet_slot_config);

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
    // workers_per_type = num_workers_per_link / 2
    const uint32_t workers_per_type = num_workers_per_link / 2;

    // Two-semaphore design: each forwarder instance has R1 and R2 semaphores
    // Slots per worker: 1 MS + num_l_chunks L chunks = (1 + num_l_chunks)
    // Slots per semaphore (R1 or R2): workers_per_type * (1 + num_l_chunks)
    const uint32_t slots_per_worker = 1 + num_l_chunks;
    const uint32_t slots_per_round = workers_per_type * slots_per_worker;

    // Validate semaphore bit capacity
    TT_FATAL(
        slots_per_round <= 32,
        "ReduceToAll: slots_per_round ({} = {} workers * {} slots/worker) exceeds 32-bit semaphore capacity",
        slots_per_round,
        workers_per_type,
        slots_per_worker);

    // Use first 2 cores from input_forwarder_cores (renamed to forwarder cores) or default
    std::vector<CoreCoord> forwarder_cores = {CoreCoord(2, 0), CoreCoord(2, 1)};
    if (operation_attributes.input_forwarder_cores.has_value() &&
        operation_attributes.input_forwarder_cores.value().size() == 2) {
        forwarder_cores = {
            operation_attributes.input_forwarder_cores.value()[0],
            operation_attributes.input_forwarder_cores.value()[1]};
    }
    CoreRangeSet forwarder_core_range_set = CoreRangeSet(forwarder_cores);

    // forwarder buffer layout per core (shared by BRISC and NCRISC):
    // - BRISC region (offset 0): slots for FWD direction
    //   - R1 slots: 0 to slots_per_round-1
    //   - R2 slots: slots_per_round to 2*slots_per_round-1 (separate region for streaming overlap)
    // - NCRISC region: same layout, offset after BRISC
    // Each slot = packet header + chunk payload (slot_size already computed and L1-aligned above)
    // R1 and R2 use SEPARATE buffer regions to support streaming (R2 can start while R1 forwarding)
    const uint32_t r2_buffer_offset = slots_per_round * slot_size;       // R2 slots start after R1 slots
    const uint32_t brisc_buffer_size = 2 * slots_per_round * slot_size;  // R1 + R2 slots
    const uint32_t ncrisc_buffer_offset = brisc_buffer_size;  // NCRISC starts after BRISC region
    const uint32_t total_forwarder_buffer_size = 2 * brisc_buffer_size;  // BRISC + NCRISC regions

    // Allocate forwarder buffer in L1 on forwarder cores
    // Use scratch tensor if provided (preferred - decouples from semaphore allocations)
    // Otherwise fall back to l1_unreserved_base_address (legacy behavior)
    uint32_t forwarder_buffer_base = 0;
    if (tensor_args.optional_forwarder_scratch_tensor.has_value()) {
        const auto& scratch_tensor = tensor_args.optional_forwarder_scratch_tensor.value();
        const uint32_t scratch_size = scratch_tensor.buffer()->size();
        TT_FATAL(
            scratch_size >= total_forwarder_buffer_size,
            "ReduceToAll: forwarder scratch tensor size ({} bytes) is insufficient. "
            "Need {} bytes = {} slots * {} slot_size * 2 rounds (R1+R2) * 2 directions",
            scratch_size,
            total_forwarder_buffer_size,
            slots_per_round,
            slot_size);
        forwarder_buffer_base = scratch_tensor.buffer()->address();
    } else {
        const uint32_t l1_unreserved_base_address =
            mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        forwarder_buffer_base = l1_unreserved_base_address;
    }

    // Create semaphores for forwarder synchronization (4 per forwarder core)
    // Two-semaphore design: each forwarder instance has R1 and R2 semaphores
    // Non-blocking design: Each semaphore is bit-packed (1 << slot_idx) per worker.
    // forwarder polls both semaphores, forwards ready slots immediately.
    // Layout: [link0_fwd_r1, link0_fwd_r2, link0_bwd_r1, link0_bwd_r2, link1_fwd_r1, ...]
    // Maximum 32 slots per semaphore (limited by 32-bit width).
    std::vector<uint32_t> forwarder_sem_addrs;
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        CoreCoord agg_core = forwarder_cores[link_idx];
        forwarder_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // fwd_r1
        forwarder_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // fwd_r2
        forwarder_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // bwd_r1
        forwarder_sem_addrs.push_back(CreateSemaphore(program, {agg_core}, 0));  // bwd_r2
    }

    // forwarder compile-time args (two-semaphore design)
    // forwarder polls both R1 and R2 semaphores, forwards any ready slots immediately.
    // R1 and R2 use separate buffer regions to support streaming overlap.
    std::vector<uint32_t> forwarder_ct_args = {
        slots_per_round,   // 0: Slots per semaphore (R1 or R2)
        slot_size,         // 1: Max packet size (header + payload)
        r2_buffer_offset,  // 2: Offset from buffer base for R2 slots
    };

    // =========================================================================
    // Create kernel compile-time args
    // =========================================================================

    // Reader compile-time args
    // Buffer layout: [L_chunk0][L_chunk1]...[MS] (L at offset 0, MS at end)
    std::vector<uint32_t> reader_ct_args = {
        cb_local_l,          // 0
        cb_local_ms,         // 1
        cb_r1_neighbor_l,    // 2
        cb_r1_neighbor_ms,   // 3
        cb_r2_neighbor_l,    // 4
        cb_r2_neighbor_ms,   // 5
        Sq_chunk_t,          // 6
        vDHt,                // 7
        ms_tile_size_bytes,  // 8 - MS tile size
        l_chunk_size_bytes,  // 9 - L chunk size
        num_l_chunks,        // 10 - number of L chunks
        tiles_per_l_chunk,   // 11 - tiles per L chunk
    };

    // Writer compile-time args
    // Buffer layout: [L_chunk0][L_chunk1]...[MS] (L at offset 0, MS at end)
    std::vector<uint32_t> writer_ct_args = {
        cb_local_l,             // 0
        cb_local_ms,            // 1
        cb_r1_result_l,         // 2
        cb_r1_result_ms,        // 3
        cb_packet_slot,         // 4: Unified packet slot CB (header + payload)
        l1_alignment,           // 5
        input_page_size_bytes,  // 6
        slot_size,              // 7: forwarder slot size (L1-aligned)
        ms_tile_size_bytes,     // 8: MS tile size
        l_chunk_size_bytes,     // 9: L chunk size
        num_l_chunks,           // 10: number of L chunks
        tiles_per_l_chunk,      // 11: tiles per L chunk
    };

    // Compute compile-time args
    std::vector<uint32_t> compute_ct_args = {
        cb_local_l,         // 0
        cb_local_ms,        // 1
        cb_r1_neighbor_l,   // 2
        cb_r1_neighbor_ms,  // 3
        cb_r1_result_l,     // 4
        cb_r1_result_ms,    // 5
        cb_r2_neighbor_l,   // 6
        cb_r2_neighbor_ms,  // 7
        cb_l_out,           // 8 - ALIASED to output_l
        cb_ms_out,          // 9 - intermediate MS
        scale_val,          // 10
        block_size,         // 11 - block_size (= tiles_per_l_chunk, max 8)
        num_blocks,         // 12 - num_blocks (= num_l_chunks)
        num_l_chunks,       // 13 - number of L chunks
        tiles_per_l_chunk,  // 14 - tiles per L chunk
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
    // Two-semaphore design: each forwarder instance has R1 and R2 semaphores
    const auto src_node_id = mesh_device->get_fabric_node_id(device_coordinate);
    for (uint32_t link_idx = 0; link_idx < num_links; link_idx++) {
        CoreCoord agg_core = forwarder_cores[link_idx];

        // Semaphore addresses for this forwarder core (4 per core: fwd_r1, fwd_r2, bwd_r1, bwd_r2)
        uint32_t fwd_r1_sem = forwarder_sem_addrs[link_idx * 4 + 0];
        uint32_t fwd_r2_sem = forwarder_sem_addrs[link_idx * 4 + 1];
        uint32_t bwd_r1_sem = forwarder_sem_addrs[link_idx * 4 + 2];
        uint32_t bwd_r2_sem = forwarder_sem_addrs[link_idx * 4 + 3];

        // BRISC runtime args (FWD direction)
        // buffer_offset = 0 (BRISC uses first region)
        std::vector<uint32_t> brisc_rt_args = {
            forwarder_buffer_base,  // 0: buffer_base
            0,                      // 1: buffer_offset (BRISC uses offset 0)
            fwd_r1_sem,             // 2: R1 semaphore
            fwd_r2_sem,             // 3: R2 semaphore
        };
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_node_id, forward_node_id, link_idx, program, agg_core, brisc_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, forwarder_brisc_kernel, agg_core, brisc_rt_args);

        // NCRISC runtime args (BWD direction)
        // buffer_offset = ncrisc_buffer_offset (NCRISC uses second region)
        std::vector<uint32_t> ncrisc_rt_args = {
            forwarder_buffer_base,  // 0: buffer_base
            ncrisc_buffer_offset,   // 1: buffer_offset (NCRISC uses offset after BRISC region)
            bwd_r1_sem,             // 2: R1 semaphore
            bwd_r2_sem,             // 3: R2 semaphore
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

        // Semaphore addresses for this forwarder core (4 per core: fwd_r1, fwd_r2, bwd_r1, bwd_r2)
        uint32_t fwd_r1_sem = forwarder_sem_addrs[link_idx * 4 + 0];
        uint32_t fwd_r2_sem = forwarder_sem_addrs[link_idx * 4 + 1];
        uint32_t bwd_r1_sem = forwarder_sem_addrs[link_idx * 4 + 2];
        uint32_t bwd_r2_sem = forwarder_sem_addrs[link_idx * 4 + 3];

        // Track slot indices for R1 and R2 within each semaphore
        // Each worker gets slots_per_worker consecutive slots (1 MS + num_l_chunks L)
        // Separate indices for FWD vs BWD direction
        uint32_t fwd_r1_worker_idx = 0;  // FWD direction R1 worker count
        uint32_t fwd_r2_worker_idx = 0;  // FWD direction R2 worker count
        uint32_t bwd_r1_worker_idx = 0;  // BWD direction R1 worker count
        uint32_t bwd_r2_worker_idx = 0;  // BWD direction R2 worker count

        for (uint32_t worker_idx = 0; worker_idx < cores_for_link.size(); worker_idx++) {
            CoreCoord core = cores_for_link[worker_idx];
            auto core_noc = mesh_device->worker_core_from_logical_core(core);

            // Determine worker type: Type A if (device_id + core_index) % 2 == 0
            // Type A: R1=FWD, R2=BWD; Type B: R1=BWD, R2=FWD
            const bool is_type_a = ((device_index + worker_idx) % 2) == 0;

            // Assign BASE slot indices within each semaphore's slot space
            // Each worker gets slots_per_worker consecutive slots (1 MS + num_l_chunks L)
            // R1 and R2 use separate semaphores but share buffer space
            uint32_t r1_slot_idx;  // Base slot for R1 (MS=slot 0, L=slots 1..num_l_chunks)
            uint32_t r2_slot_idx;  // Base slot for R2
            uint32_t r1_agg_sem;
            uint32_t r2_agg_sem;
            auto r1_dst_node_id = forward_node_id;   // Default to forward
            auto r2_dst_node_id = backward_node_id;  // Default to backward

            if (is_type_a) {
                // Type A: R1=FWD, R2=BWD
                r1_slot_idx = fwd_r1_worker_idx * slots_per_worker;
                fwd_r1_worker_idx++;
                r2_slot_idx = bwd_r2_worker_idx * slots_per_worker;
                bwd_r2_worker_idx++;
                r1_agg_sem = fwd_r1_sem;
                r2_agg_sem = bwd_r2_sem;
                r1_dst_node_id = forward_node_id;
                r2_dst_node_id = backward_node_id;
            } else {
                // Type B: R1=BWD, R2=FWD
                r1_slot_idx = bwd_r1_worker_idx * slots_per_worker;
                bwd_r1_worker_idx++;
                r2_slot_idx = fwd_r2_worker_idx * slots_per_worker;
                fwd_r2_worker_idx++;
                r1_agg_sem = bwd_r1_sem;
                r2_agg_sem = fwd_r2_sem;
                r1_dst_node_id = backward_node_id;
                r2_dst_node_id = forward_node_id;
            }

            // Calculate forwarder slot addresses based on direction
            // R1 and R2 use SEPARATE buffer regions to support streaming overlap
            // R1 slots: buffer_base + slot_idx * slot_size
            // R2 slots: buffer_base + r2_buffer_offset + slot_idx * slot_size
            uint32_t r1_slot_base = is_type_a ? forwarder_buffer_base : (forwarder_buffer_base + ncrisc_buffer_offset);
            uint32_t r2_slot_base = is_type_a ? (forwarder_buffer_base + ncrisc_buffer_offset) : forwarder_buffer_base;

            // Calculate actual slot addresses (R2 includes r2_buffer_offset for separate region)
            uint32_t r1_slot_addr = r1_slot_base + (r1_slot_idx * slot_size);
            uint32_t r2_slot_addr = r2_slot_base + r2_buffer_offset + (r2_slot_idx * slot_size);

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
                *r1_dst_node_id.mesh_id,             // 0: R1 neighbor destination mesh ID
                r1_dst_node_id.chip_id,              // 1: R1 neighbor destination chip ID
                r1_recv_tensor.buffer()->address(),  // 2: R1 neighbor destination
                r1_recv_sem.address(),               // 3: R1 neighbor semaphore
                *r2_dst_node_id.mesh_id,             // 4: R2 neighbor destination mesh ID
                r2_dst_node_id.chip_id,              // 5: R2 neighbor destination chip ID
                r2_recv_tensor.buffer()->address(),  // 6: R2 neighbor destination
                r2_recv_sem.address(),               // 7: R2 neighbor semaphore
                core_noc.x,                          // 8: current_core_x
                core_noc.y,                          // 9: current_core_y
                // forwarder-specific args
                agg_core_noc.x,  // 10: forwarder_core_x
                agg_core_noc.y,  // 11: forwarder_core_y
                r1_slot_addr,    // 12: R1 forwarder slot address
                r1_agg_sem,      // 13: R1 forwarder semaphore
                r1_slot_idx,     // 14: R1 BASE slot index (chunk i signals: 1 << (base + i))
                r2_slot_addr,    // 15: R2 forwarder slot address
                r2_agg_sem,      // 16: R2 forwarder semaphore
                r2_slot_idx,     // 17: R2 BASE slot index (chunk i signals: 1 << (base + i))
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
            // 0: R1 mesh_id (static), 1: R1 chip_id (static)
            // 2: R1 dest addr, 3: R1 sem addr
            // 4: R2 mesh_id (static), 5: R2 chip_id (static)
            // 6: R2 dest addr, 7: R2 sem addr
            // 8-17: forwarder args (static)
            auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_variables.writer_kernel);
            auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
            writer_runtime_args[2] = intermediate_tensors[0].buffer()->address();  // R1 dest
            writer_runtime_args[3] = shared_variables.semaphores[0].address();     // R1 sem
            // Indices 4-5 (mesh_id, chip_id) are static - don't update
            writer_runtime_args[6] = intermediate_tensors[1].buffer()->address();  // R2 dest
            writer_runtime_args[7] = shared_variables.semaphores[1].address();     // R2 sem
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
