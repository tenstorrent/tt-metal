// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include <tt-metalium/constants.hpp>
#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/zero_init_common.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/combine_sf.hpp"

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_COMBINE(...)
#endif

// Signal last element to writer to break out of loop
constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-4)
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dispatched_metadata_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(4);

    // Page counts (indices 5-8)
    constexpr uint32_t dispatched_buffer_pages = get_compile_time_arg_val(5);
    constexpr uint32_t dispatched_metadata_pages = get_compile_time_arg_val(6);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(7);
    constexpr uint32_t output_pages = get_compile_time_arg_val(8);

    // Page sizes (indices 9-12)
    constexpr uint32_t dispatched_buffer_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t dispatched_metadata_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t experts_tok_counter_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(12);

    // Operation parameters (indices 13-16)
    constexpr uint32_t num_chips = get_compile_time_arg_val(13);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(14);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(15);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(16);

    // Hidden dimension (index 17)
    constexpr uint32_t hidden_size = get_compile_time_arg_val(17);

    // Aligned page sizes (indices 18-21)
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(21);

    // Mesh information (indices 22-26)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(22);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(23);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(24);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(25);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(26);

    // Fabric configuration (indices 27-30)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(27);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(28);
    constexpr uint32_t num_links = get_compile_time_arg_val(29);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(30);

    // Batch configuration (index 31)
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(31);
    // Number of dispatch groups (index 32)
    constexpr uint32_t num_dispatch_groups = get_compile_time_arg_val(32);

    // Expert region offsets tensor metadata (indices 33-36)
    constexpr uint32_t cb_expert_region_offsets_id = get_compile_time_arg_val(33);
    constexpr uint32_t expert_region_offsets_pages = get_compile_time_arg_val(34);
    constexpr uint32_t expert_region_offsets_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_expert_region_offsets_page_size = get_compile_time_arg_val(36);

    // Dispatch buffer total per-chip capacity (index 37) — used as overflow guard.
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(37);

    // TensorAccessorArgs for all 5 tensors (starting at index 38)
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<38>();
    constexpr auto dispatched_metadata_args =
        TensorAccessorArgs<dispatched_buffer_args.next_compile_time_args_offset()>();
    constexpr auto experts_tok_counter_args =
        TensorAccessorArgs<dispatched_metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<experts_tok_counter_args.next_compile_time_args_offset()>();
    constexpr auto expert_region_offsets_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

#if USE_STORE_AND_FORWARD
    constexpr auto staging_args = TensorAccessorArgs<expert_region_offsets_args.next_compile_time_args_offset()>();
    constexpr uint32_t after_accessor_args = staging_args.next_compile_time_args_offset();
#else
    constexpr uint32_t after_accessor_args = expert_region_offsets_args.next_compile_time_args_offset();
#endif

#if INIT_ZEROS
    // Zero-init args follow immediately after the TensorAccessorArgs block
    constexpr uint32_t cb_zero_buffer_id = get_compile_time_arg_val(after_accessor_args);
    constexpr uint32_t num_total_untilizer_cores = get_compile_time_arg_val(after_accessor_args + 1);
    constexpr uint32_t tile_layout_args_base = after_accessor_args + 2;
#else
    constexpr uint32_t tile_layout_args_base = after_accessor_args;
#endif

    // Sender always consumes untilized rows + routing metadata from its dedicated untilizer
    // group's receive_buf (c_18) / metadata ring (c_19); these args are appended per-sender by
    // the program factory for both TILE_LAYOUT and ROW_MAJOR (the ROW_MAJOR untilizer reader
    // page-copies rows into c_2 instead of untilizing, but the sender path is identical).
    constexpr uint32_t num_untilizer_cores_group = get_compile_time_arg_val(tile_layout_args_base);
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(tile_layout_args_base + 1);
    constexpr uint32_t cb_metadata_buf_id = get_compile_time_arg_val(tile_layout_args_base + 2);
    // Per-untilizer ring depth on the sender's receive_buf (drives the slot ring below).
    constexpr uint32_t SLOTS_PER_UNTILIZER = get_compile_time_arg_val(tile_layout_args_base + 3);

#if USE_STORE_AND_FORWARD
    namespace sf = ttnn::operations::experimental::deepseek_prefill::combine::sf;
    // Which of this chip's sender cores we are; selects our slice of every staging ring.
    constexpr uint32_t sf_my_core = get_compile_time_arg_val(tile_layout_args_base + 4);
    constexpr uint32_t sf_levels = SF_LEVELS;
    constexpr uint32_t sf_slots = SF_SLOTS;
    constexpr uint32_t sf_hdr_bytes = SF_HDR_BYTES;
    constexpr uint32_t sf_num_cores = SF_NUM_CORES;
    constexpr uint32_t sf_out_live_mask[2] = SF_OUT_LIVE_MASK;
    constexpr uint32_t sf_in_live_mask[2] = SF_IN_LIVE_MASK;
    constexpr uint32_t sf_neighbour[2] = SF_NEIGHBOUR;
    // Serve at most this many relayed pages before giving an injection a turn.  Transit outranks
    // injection because a downstream chip is already waiting on it, but injection carries the
    // untilizer sentinels that end-of-stream is derived from, so it cannot be starved outright.
    constexpr uint32_t SF_INJ_QUANTUM = 8;
    auto sf_out_live = [&](uint32_t d, uint32_t r) { return ((sf_out_live_mask[d] >> (r - 1)) & 1u) != 0; };
    auto sf_in_live = [&](uint32_t d, uint32_t r) { return ((sf_in_live_mask[d] >> (r - 1)) & 1u) != 0; };
#endif

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatched_metadata_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t experts_tok_counter_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_region_offsets_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_init_complete_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_init_barrier_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t num_cores = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_args++);

    DPRINT_COMBINE(
        "Combine Reader: experts=[{}, {}) linearized_mesh_coord={}\n",
        expert_start_idx,
        expert_end_idx,
        linearized_mesh_coord);

#if INIT_ZEROS
    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

    // Hybrid row output-zeroing: this core zeroes its assigned page range, then waits for untilizer row cores
    {
        uint32_t page_start = get_arg_val<uint32_t>(rt_args++);
        uint32_t page_end = get_arg_val<uint32_t>(rt_args++);
        uint32_t output_init_done_semaphore_id = get_arg_val<uint32_t>(rt_args++);

        {
            // DeviceZoneScopedN("combine-output-zeroing-SENDER-writing");
            zero_pages(cb_zero_buffer_id, page_start, page_end, aligned_output_page_size, output_addr_gen);
        }

        Semaphore<> output_init_done_sem(output_init_done_semaphore_id);
        output_init_done_sem.wait(num_total_untilizer_cores);
        output_init_done_sem.set(0);
    }
#endif

    uint32_t counter_ready_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_start_x = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_start_y = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_end_x = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_end_y = get_arg_val<uint32_t>(rt_args++);
    uint32_t untilizer_counter_l1_offset = get_write_ptr(cb_dispatched_metadata_id);
    uint64_t mcast_counter_noc_addr =
        get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, untilizer_counter_l1_offset);
    Semaphore<> counter_ready_sem(counter_ready_semaphore_id);
    Noc noc_obj;

    // Per-untilizer semaphores (each scoped to just the (this sender, untilizer) pair):
    //   data_ready: untilizer ++ after each non-local row it writes into receive_buf.  We do
    //               wait(>=1) + atomic dec(-1) to consume exactly one per row.
    //   credits:    init SLOTS_PER_UNTILIZER on untilizer's L1; we ++ untilizer's copy each time we free
    //               a row slot in its 16-deep ring on our receive_buf.
    // Both the wait-side L1 ptr (on this sender) and the inc-side NOC address (on untilizer)
    // refer to the same logical sem; pair-scoped allocation guarantees the L1 offset is
    // identical on both cores.
    volatile tt_l1_ptr uint32_t* data_ready_sem_ptrs[num_untilizer_cores_group];
    [[maybe_unused]] uint64_t self_data_ready_noc_addrs[num_untilizer_cores_group];
    uint64_t untilizer_credits_noc_addrs[num_untilizer_cores_group];
    uint32_t untilizer_noc_x[num_untilizer_cores_group];
    uint32_t untilizer_noc_y[num_untilizer_cores_group];
    for (uint32_t c = 0; c < num_untilizer_cores_group; c++) {
        uint32_t data_ready_semaphore_id = get_arg_val<uint32_t>(rt_args++);
        uint32_t credits_semaphore_id = get_arg_val<uint32_t>(rt_args++);
        untilizer_noc_x[c] = get_arg_val<uint32_t>(rt_args++);
        untilizer_noc_y[c] = get_arg_val<uint32_t>(rt_args++);
        uint32_t data_ready_l1 = get_semaphore(data_ready_semaphore_id);
        uint32_t credits_l1 = get_semaphore(credits_semaphore_id);
        data_ready_sem_ptrs[c] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_l1);
        self_data_ready_noc_addrs[c] = get_noc_addr(my_x[noc_index], my_y[noc_index], data_ready_l1);
        untilizer_credits_noc_addrs[c] = get_noc_addr(untilizer_noc_x[c], untilizer_noc_y[c], credits_l1);
    }

#if USE_STORE_AND_FORWARD
    const uint32_t sf_staging_addr = get_arg_val<uint32_t>(rt_args++);
    const uint32_t sf_cleanup_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    // Arrival and credit counters, ordered to match the host's construction.  Both are indexed by
    // the DATA direction; a credit increment for direction d travels the opposite way, which is the
    // single most error-prone thing in this protocol.
    uint32_t sf_arrived_addr[2][sf_levels];
    uint32_t sf_credit_addr[2][sf_levels];
    for (uint32_t d = 0; d < 2; d++) {
        for (uint32_t r = 0; r < sf_levels; r++) {
            sf_arrived_addr[d][r] = get_arg_val<uint32_t>(rt_args++);
            sf_credit_addr[d][r] = get_arg_val<uint32_t>(rt_args++);
        }
    }
    const auto sf_staging_gen = TensorAccessor(staging_args, sf_staging_addr);

    // Pages we have handed to the downstream ring, versus pages it has told us it freed.  Both
    // monotonic with a single writer each, so no read-modify-write can race.
    uint32_t sf_staged[2][sf_levels] = {};
    uint32_t sf_pool_rd[2][sf_levels] = {};
    bool sf_eos_out[2][sf_levels] = {};
    uint32_t sf_cred_pending[2][sf_levels] = {};
    uint32_t sf_arrived_pending[2][sf_levels] = {};
    uint32_t sf_transit_run = 0;
    // Which page of the reader->writer queue the next push lands on.  Needed because a batch is
    // written as one contiguous run: a bulk reserve that starts near the end of the queue wraps,
    // and the wrapped pages are NOT contiguous with the first.  Single-page control and injection
    // pushes leave the write pointer on any index, so batch alignment cannot be assumed.
    uint32_t sf_cb_wr = 0;

    auto sf_arrived_raw = [&](uint32_t d, uint32_t r) {
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sf_arrived_addr[d][r - 1]);
    };
    // An arrival count carries the end-of-stream flag as an additive bias, so a close can never be
    // observed ahead of the last batched increment that preceded it.
    auto sf_closed = [&](uint32_t d, uint32_t r) { return sf_arrived_raw(d, r) >= sf::EOS_BIAS; };
    auto sf_total = [&](uint32_t d, uint32_t r) {
        const uint32_t raw = sf_arrived_raw(d, r);
        return raw >= sf::EOS_BIAS ? raw - sf::EOS_BIAS : raw;
    };
    auto sf_credit = [&](uint32_t d, uint32_t r) {
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sf_credit_addr[d][r - 1]);
    };
    // Level 0 is the destination's own output page, which is preallocated and always available.
    // That is the base of the deadlock argument: the lowest level never has to wait for anything.
    auto sf_has_room = [&](uint32_t d, uint32_t r) {
        return r == 0 || (sf_staged[d][r - 1] - sf_credit(d, r) < sf_slots);
    };
    auto sf_base_page = [&](uint32_t d, uint32_t r) {
        return sf::base_page(d, r, sf_my_core, sf_levels, sf_num_cores, sf_slots);
    };
#endif

#if INIT_ZEROS
    // Signal writer that output-zeroing is complete
    Semaphore<> output_init_complete_sem(output_init_complete_semaphore_id);
    output_init_complete_sem.set(1);

    // Wait for ALL writers (all cores) to complete init exchange.
    // Each writer signals all readers' barrier sems via noc_semaphore_inc,
    // so this reader waits for num_cores signals before proceeding.
    Semaphore<> barrier_sem(output_init_barrier_semaphore_id);
    barrier_sem.wait(num_cores);
    barrier_sem.set(0);
#endif

    // Read expert token counts
    const auto experts_tok_counter_addr_gen = TensorAccessor(experts_tok_counter_args, experts_tok_counter_addr);
    cb_reserve_back(cb_experts_tok_counter_id, experts_tok_counter_pages);
    uint32_t counter_base_addr = get_write_ptr(cb_experts_tok_counter_id);
    {
        // DeviceZoneScopedN("combine-reading-expert-token-counts");
        for (uint32_t i = 0; i < experts_tok_counter_pages; i++) {
            noc_async_read_page(
                i, experts_tok_counter_addr_gen, counter_base_addr + i * aligned_experts_tok_counter_page_size);
        }
        noc_async_read_barrier();
    }

    // Expert token counts: flat [num_routed_experts] array per device.
    // Decompose linearized_mesh_coord into (row, col) using physical mesh dims,
    // then map col -> dispatch_group_idx via modulo num_dispatch_groups.
    // This handles DP replicas (ndg < mesh_cols) where multiple columns share the same group.
    constexpr uint32_t mesh_row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t mesh_col = linearized_mesh_coord % mesh_cols;
    constexpr uint32_t dispatch_group_idx = mesh_col % num_dispatch_groups;
    constexpr uint32_t experts_per_dispatch_group = experts_per_chip * num_chips;
    constexpr uint32_t offset = dispatch_group_idx * experts_per_dispatch_group + mesh_row * experts_per_chip;
    // Multicast expert token counts + receive_buf_addr to all untilizer cores
    // Each sender multicasts token counts + its own receive_buf_addr to its dedicated untilizer
    // group. The mcast destination covers only this sender's k_s untilizer cores (per-sender
    // bounding box), so all senders can multicast in parallel.
    // Trailer layout (one l1_alignment region after counter_total_size bytes):
    //   [0]: receive_buf_addr  — sender's c_18 L1 offset (where untilizer NOC-writes untilized data)
    //   [1]: metadata_buf_addr — sender's c_19 L1 offset (where untilizer NOC-writes routing metadata)
    {
        // DeviceZoneScopedN("combine-sender-multicast-sending");
        constexpr uint32_t counter_total_size = experts_tok_counter_pages * aligned_experts_tok_counter_page_size;

        volatile tt_l1_ptr uint32_t* trailer_slot =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_base_addr + counter_total_size);
        trailer_slot[0] = get_write_ptr(cb_untilize_id);
        trailer_slot[1] = get_write_ptr(cb_metadata_buf_id);

        constexpr uint32_t mcast_total_size = counter_total_size + l1_alignment;
        uint32_t off = 0;
        while (off < mcast_total_size) {
            uint32_t chunk = ((mcast_total_size - off) > (uint32_t)NOC_MAX_BURST_SIZE) ? (uint32_t)NOC_MAX_BURST_SIZE
                                                                                       : (mcast_total_size - off);
            noc_async_write_multicast(
                counter_base_addr + off, mcast_counter_noc_addr + off, chunk, num_untilizer_cores_group);
            off += chunk;
        }
        noc_async_write_barrier();
        counter_ready_sem.inc_multicast(
            noc_obj, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, /*value=*/1, num_untilizer_cores_group);
    }

    uint32_t untilize_base = get_write_ptr(cb_untilize_id);
    uint32_t metadata_buf_base = get_write_ptr(cb_metadata_buf_id);
    // Both receive_buf (c_18) and metadata_ring (c_19) are partitioned k_s ways:
    // untilizer c owns a SLOTS_PER_UNTILIZER-deep ring starting at
    //   untilize_base    + c * SLOTS_PER_UNTILIZER * aligned_output_page_size
    //   metadata_buf_base + c * SLOTS_PER_UNTILIZER * aligned_dispatched_metadata_page_size
    // read_slots[c] tracks the next slot index (mod SLOTS_PER_UNTILIZER) to pull from for untilizer c.
    uint32_t read_slots[num_untilizer_cores_group];
    for (uint32_t c = 0; c < num_untilizer_cores_group; c++) {
        read_slots[c] = 0;
    }

    // Round-robin polling loop — sender polls all untilizer core CBs without blocking on any
    // single one.  Each untilizer core writes routing metadata + row data for every non-local row,
    // then sends ROUTE_INFO_SENTINEL when all its batches are complete.  Sender exits when
    // every untilizer core has signalled done, eliminating head-of-line blocking between cores.
    {
        static_assert(
            (SLOTS_PER_UNTILIZER & (SLOTS_PER_UNTILIZER - 1)) == 0, "SLOTS_PER_UNTILIZER must be a power of 2");
        constexpr uint32_t SLOTS_PER_UNTILIZER_MASK = SLOTS_PER_UNTILIZER - 1;

        uint32_t untilizer_done_count = 0;
        bool untilizer_finished[num_untilizer_cores_group];
        // consumed[c] tracks how many data_ready increments we've processed for untilizer core c.
        // The untilizer core only ever INCREMENTS data_ready_sem; the sender never decrements it.
        // Replaces the per-row noc_semaphore_inc(-1) + noc_async_atomic_barrier round-trip
        // with a local register-resident counter compare.
        uint32_t consumed[num_untilizer_cores_group];
        uint32_t ring_meta_addr[num_untilizer_cores_group][SLOTS_PER_UNTILIZER];
        uint64_t buffer_scratch_noc_addr_table[num_untilizer_cores_group][SLOTS_PER_UNTILIZER];
        for (uint32_t c = 0; c < num_untilizer_cores_group; c++) {
            untilizer_finished[c] = false;
            consumed[c] = 0;
            uint32_t meta_addr = metadata_buf_base + c * SLOTS_PER_UNTILIZER * aligned_dispatched_metadata_page_size;
            uint32_t out_addr = untilize_base + c * SLOTS_PER_UNTILIZER * aligned_output_page_size;
            for (uint32_t s = 0; s < SLOTS_PER_UNTILIZER; s++) {
                ring_meta_addr[c][s] = meta_addr;
                buffer_scratch_noc_addr_table[c][s] = get_noc_addr(out_addr);
                meta_addr += aligned_dispatched_metadata_page_size;
                out_addr += aligned_output_page_size;
            }
        }

#if USE_STORE_AND_FORWARD
        // Route toward each axis neighbour, resolved once.  A token's direction is identified by
        // matching its own first-hop route against these, which avoids duplicating the topology
        // arithmetic that get_route already owns.
        constexpr uint32_t SF_ROUTE_NONE = 0xFFu;
        uint32_t sf_dir_route[2];
        for (uint32_t d = 0; d < 2; d++) {
            sf_dir_route[d] = sf_neighbour[d] == SF_NO_NEIGHBOUR
                                  ? SF_ROUTE_NONE
                                  : get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, sf_neighbour[d]);
        }

        constexpr uint32_t SF_SLOT_MASK = sf_slots - 1;
        constexpr uint32_t SF_BUMP_EVERY = sf::BUMP_EVERY;

        // Control-only queue entries carry no payload but still take a slot, which is why the queue
        // is deeper than the two the non-relaying path needs.
        auto sf_push_ctl = [&](uint32_t cmd, uint32_t d, uint32_t level, uint32_t sem_addr, uint32_t value) {
            cb_reserve_back(cb_route_info_id, 1);
            volatile tt_l1_ptr uint32_t* hdr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
            hdr[sf::HDR_ROUTE] = sf_dir_route[d];
            hdr[sf::HDR_DISTANCE] = 1;
            hdr[sf::HDR_PAGE_IDX] = 0;
            hdr[sf::HDR_DST_CHIP] = sf_neighbour[d];
            hdr[sf::HDR_CMD] = cmd;
            hdr[sf::HDR_SEM_ADDR] = sem_addr;
            hdr[sf::HDR_INC_VALUE] = value;
            hdr[sf::HDR_INC_DIR] = d;
            cb_push_back(cb_route_info_id, 1);
            sf_cb_wr = (sf_cb_wr + 1) % SF_RING_DEPTH;
        };

        // An arrival travels with the data it accounts for; a credit travels back against it, so it
        // is addressed to the upstream neighbour, which is the one in the OPPOSITE direction.
        auto sf_flush_counters = [&](bool force) {
            for (uint32_t d = 0; d < 2; d++) {
                for (uint32_t r = 1; r <= sf_levels; r++) {
                    uint32_t& arrived = sf_arrived_pending[d][r - 1];
                    if (arrived != 0 && (force || arrived >= SF_BUMP_EVERY)) {
                        sf_push_ctl(sf::CMD_ARRIVED_INC, d, r, sf_arrived_addr[d][r - 1], arrived);
                        arrived = 0;
                    }
                    uint32_t& credit = sf_cred_pending[d][r - 1];
                    if (credit != 0 && (force || credit >= SF_BUMP_EVERY)) {
                        sf_push_ctl(sf::CMD_CREDIT_INC, 1 - d, r, sf_credit_addr[d][r - 1], credit);
                        credit = 0;
                    }
                }
            }
        };

        auto sf_drained = [&](uint32_t d, uint32_t r) {
            return sf_closed(d, r) && sf_pool_rd[d][r - 1] == sf_total(d, r);
        };

        // Relay up to SF_BATCH pages one hop.  Which FIFO a page came out of tells us everything:
        // the direction, the outbound FIFO, and whether this hop is the last one.
        //
        // Every read is issued before any of them is waited on.  A read-then-barrier per page leaves
        // the reader idle for the whole of each page's DRAM latency, and that is measurably the
        // binding constraint -- the reader sits at 100% of the op's kernel duration while compute
        // idles and DRAM bandwidth is far from saturated.
        //
        // Returns the number of pages moved, not a flag: the caller's fairness quantum counts relay
        // steps against injections, so reporting a batch as a single step would let one relay pass
        // starve the untilizers by up to SF_BATCH times as long.
        constexpr uint32_t SF_BATCH_N = SF_BATCH;
        constexpr uint32_t sf_queue_stride = sf_hdr_bytes + SF_PAGE_BYTES;
        auto sf_try_transit = [&]() -> uint32_t {
            uint32_t picked_d[SF_BATCH_N];
            uint32_t picked_r[SF_BATCH_N];
            uint32_t picked_page[SF_BATCH_N];
            uint32_t n = 0;

            // Never gather more pages than the queue can already take.  Blocking for a whole batch
            // deadlocks: the writer frees these slots, but the writer can be waiting on a credit or
            // arrival increment that only this reader emits -- and emitting one needs a slot of its
            // own.  Capping at what is reservable right now keeps the single-page fallback, whose
            // liveness the one-at-a-time path already relied on.
            // Stop the batch at the wrap: past it the pages are not contiguous with the first.
            uint32_t batch_cap = SF_RING_DEPTH - sf_cb_wr;
            if (batch_cap > SF_BATCH_N) {
                batch_cap = SF_BATCH_N;
            }
            while (batch_cap > 1 && !cb_pages_reservable_at_back(cb_route_info_id, batch_cap)) {
                batch_cap--;
            }

            // No cross-pick accounting is needed here, and that is a property of the loop shape
            // rather than luck: each (direction, level) is visited once per pass and yields at most
            // one page, so every pick in a batch comes from a distinct source FIFO and targets a
            // distinct downstream FIFO.  Two picks can therefore never contend for one source slot
            // or one destination slot.  Widening this loop to take several pages from one FIFO would
            // break that and would need explicit per-pick slot accounting -- which is exactly the
            // kind of bug that keeps the traffic volume right and silently corrupts the contents.
            for (uint32_t d = 0; d < 2 && n < batch_cap; d++) {
                for (uint32_t r = 1; r <= sf_levels && n < batch_cap; r++) {
                    if (!sf_in_live(d, r) || sf_pool_rd[d][r - 1] == sf_total(d, r)) {
                        continue;
                    }
                    if (!sf_has_room(d, r - 1)) {
                        continue;  // downstream full: try another source rather than wait
                    }
                    picked_d[n] = d;
                    picked_r[n] = r;
                    picked_page[n] = sf_base_page(d, r) + (sf_pool_rd[d][r - 1] & SF_SLOT_MASK);
                    n++;
                }
            }
            if (n == 0) {
                return 0;
            }

            // One reserve for the whole batch.  These n pages are a contiguous run from the write
            // pointer because batch_cap above stopped short of the wrap: a bulk reserve that crosses
            // it hands back pages that are not adjacent to the first, and writing them as a run
            // scribbles past the end of the queue.
            cb_reserve_back(cb_route_info_id, n);
            const uint32_t batch_base = get_write_ptr(cb_route_info_id);
            for (uint32_t i = 0; i < n; i++) {
                noc_async_read(
                    sf_staging_gen.get_noc_addr(picked_page[i]),
                    batch_base + i * sf_queue_stride + sf_hdr_bytes,
                    aligned_output_page_size + sf::tail_bytes());
            }
            noc_async_read_barrier();

            for (uint32_t i = 0; i < n; i++) {
                const uint32_t d = picked_d[i];
                const uint32_t r = picked_r[i];
                const uint32_t cb_base = batch_base + i * sf_queue_stride;
                const uint32_t payload = cb_base + sf_hdr_bytes;

                volatile tt_l1_ptr uint32_t* tail =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload + aligned_output_page_size);
                const uint32_t final_chip = tail[sf::TAIL_FINAL_DST_CHIP];
                // Reading an unwritten page means the arrival accounting is off by one, which
                // would otherwise show up as a single wrong token rather than as a failure.
                ASSERT(tail[sf::TAIL_MAGIC] == sf::MAGIC);
                // The level a page sits at must equal its real remaining distance.  Checking the
                // invariant beats checking a copy of it that every hop would have to re-stamp.
                // Hoisted out of ASSERT: the macro cannot carry the template argument commas.
                const uint32_t observed_hops =
                    manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, final_chip);
                ASSERT(observed_hops == r);

                volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base);
                hdr[sf::HDR_ROUTE] = sf_dir_route[d];
                hdr[sf::HDR_DISTANCE] = 1;
                hdr[sf::HDR_DST_CHIP] = sf_neighbour[d];
                hdr[sf::HDR_INC_DIR] = d;
                if (r == 1) {
                    hdr[sf::HDR_CMD] = sf::CMD_FINAL_WRITE;
                    hdr[sf::HDR_PAGE_IDX] = tail[sf::TAIL_OUTPUT_PAGE_IDX];
                } else {
                    hdr[sf::HDR_CMD] = sf::CMD_STAGE;
                    hdr[sf::HDR_PAGE_IDX] = sf_base_page(d, r - 1) + (sf_staged[d][r - 2] & SF_SLOT_MASK);
                    sf_staged[d][r - 2]++;
                    sf_arrived_pending[d][r - 2]++;
                }
                sf_pool_rd[d][r - 1]++;
                sf_cred_pending[d][r - 1]++;
            }
            cb_push_back(cb_route_info_id, n);
            sf_cb_wr = (sf_cb_wr + n) % SF_RING_DEPTH;
            return n;
        };

        // Peek the head of each untilizer ring and only commit once the destination has room.  A row
        // left uncommitted keeps its slot and its credit, which is the backpressure that stops the
        // untilizers running ahead of the network.
        auto sf_try_inject = [&]() {
            for (uint32_t c = 0; c < num_untilizer_cores_group; c++) {
                if (untilizer_finished[c]) {
                    continue;
                }
                invalidate_l1_cache();
                if (*data_ready_sem_ptrs[c] == consumed[c]) {
                    continue;
                }
                const uint32_t slot = read_slots[c];
                volatile tt_l1_ptr uint32_t* ring_meta =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring_meta_addr[c][slot]);
                const uint32_t meta0 = ring_meta[0];

                if (meta0 == ROUTE_INFO_SENTINEL) {
                    consumed[c]++;
                    read_slots[c] = (slot + 1) & SLOTS_PER_UNTILIZER_MASK;
                    noc_semaphore_set(data_ready_sem_ptrs[c], 0);
                    noc_async_atomic_barrier();
                    untilizer_finished[c] = true;
                    untilizer_done_count++;
                    return true;
                }

                const uint32_t dst_chip = meta0;
                const uint32_t distance =
                    manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);
                const uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);
                const uint32_t d = route == sf_dir_route[0] ? 0u : 1u;
                const uint32_t target_level = distance - 1;  // zero means straight to the output page
                if (!sf_has_room(d, target_level)) {
                    continue;  // no commit: the row and its credit stay put
                }

                consumed[c]++;
                read_slots[c] = (slot + 1) & SLOTS_PER_UNTILIZER_MASK;
                const uint32_t output_page_idx = ring_meta[1] * num_experts_per_tok + ring_meta[2];

                cb_reserve_back(cb_route_info_id, 1);
                const uint32_t cb_base = get_write_ptr(cb_route_info_id);
                const uint32_t payload = cb_base + sf_hdr_bytes;
                noc_async_read(buffer_scratch_noc_addr_table[c][slot], payload, aligned_output_page_size);
                noc_async_read_barrier();

                volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base);
                hdr[sf::HDR_ROUTE] = sf_dir_route[d];
                hdr[sf::HDR_DISTANCE] = 1;
                hdr[sf::HDR_DST_CHIP] = sf_neighbour[d];
                hdr[sf::HDR_INC_DIR] = d;
                if (target_level == 0) {
                    hdr[sf::HDR_CMD] = sf::CMD_FINAL_WRITE;
                    hdr[sf::HDR_PAGE_IDX] = output_page_idx;
                } else {
                    volatile tt_l1_ptr uint32_t* tail =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload + aligned_output_page_size);
                    tail[sf::TAIL_OUTPUT_PAGE_IDX] = output_page_idx;
                    tail[sf::TAIL_FINAL_DST_CHIP] = dst_chip;
                    tail[sf::TAIL_MAGIC] = sf::MAGIC;
                    tail[sf::TAIL_RESERVED] = 0;
                    hdr[sf::HDR_CMD] = sf::CMD_STAGE;
                    hdr[sf::HDR_PAGE_IDX] =
                        sf_base_page(d, target_level) + (sf_staged[d][target_level - 1] & SF_SLOT_MASK);
                    sf_staged[d][target_level - 1]++;
                    sf_arrived_pending[d][target_level - 1]++;
                }
                cb_push_back(cb_route_info_id, 1);
                sf_cb_wr = (sf_cb_wr + 1) % SF_RING_DEPTH;
                noc_semaphore_inc<true>(untilizer_credits_noc_addrs[c], 1);
                return true;
            }
            return false;
        };

        // Our downstream's level r is fed only by pages we hold of remaining distance r + 1: our own
        // injections, and our inbound level r + 1.  Level sf_levels has no inbound level above it,
        // so its close depends on nothing but local completion.  That is the base case, and it is
        // why this terminates on a ring, where waiting for an upstream close would cycle forever.
        auto sf_try_emit_eos = [&]() {
            if (untilizer_done_count < num_untilizer_cores_group) {
                return;
            }
            for (uint32_t d = 0; d < 2; d++) {
                for (uint32_t r = sf_levels; r >= 1; r--) {
                    if (!sf_out_live(d, r) || sf_eos_out[d][r - 1]) {
                        continue;
                    }
                    const bool fed_from_above = r + 1 <= sf_levels && sf_in_live(d, r + 1);
                    if (fed_from_above && !sf_drained(d, r + 1)) {
                        continue;
                    }
                    // Carry any outstanding arrivals in the same increment, so the close cannot be
                    // observed before the pages it closes over.
                    sf_push_ctl(
                        sf::CMD_ARRIVED_INC,
                        d,
                        r,
                        sf_arrived_addr[d][r - 1],
                        sf_arrived_pending[d][r - 1] + sf::EOS_BIAS);
                    sf_arrived_pending[d][r - 1] = 0;
                    sf_eos_out[d][r - 1] = true;
                }
            }
        };

        auto sf_all_done = [&]() {
            if (untilizer_done_count < num_untilizer_cores_group) {
                return false;
            }
            for (uint32_t d = 0; d < 2; d++) {
                for (uint32_t r = 1; r <= sf_levels; r++) {
                    if (sf_in_live(d, r) && !sf_drained(d, r)) {
                        return false;
                    }
                    if (sf_out_live(d, r) && !sf_eos_out[d][r - 1]) {
                        return false;
                    }
                    if (sf_cred_pending[d][r - 1] != 0 || sf_arrived_pending[d][r - 1] != 0) {
                        return false;
                    }
                }
            }
            return true;
        };

        while (!sf_all_done()) {
            bool progressed = false;
            if (sf_transit_run < SF_INJ_QUANTUM) {
                // Advance the quantum by the pages moved, not by one per pass: a batch counted as a
                // single step would let the relay hold the reader for SF_BATCH times as many pages
                // before the untilizers get a turn.
                const uint32_t moved = sf_try_transit();
                progressed = moved != 0;
                sf_transit_run += moved;
            }
            if (!progressed) {
                sf_transit_run = 0;
                progressed = sf_try_inject();
            }
            // Force the counters out when nothing moved: the peer that unblocks us is waiting on
            // exactly these, so holding a partial batch back would be the deadlock.
            sf_flush_counters(!progressed);
            sf_try_emit_eos();
        }
        sf_flush_counters(true);
#else
        while (untilizer_done_count < num_untilizer_cores_group) {
            for (uint32_t c = 0; c < num_untilizer_cores_group; c++) {
                if (untilizer_finished[c]) {
                    continue;
                }

                // Non-blocking check: data_ready lives in sender L1.  Invalidate L1 cache so the
                // load picks up any NoC-written increments from the untilizer core (the prior atomic
                // barrier used to do this for us; now we do it explicitly).
                invalidate_l1_cache();
                if (*data_ready_sem_ptrs[c] == consumed[c]) {
                    continue;
                }
                consumed[c]++;

                uint32_t slot = read_slots[c];
                volatile tt_l1_ptr uint32_t* ring_meta =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring_meta_addr[c][slot]);
                uint32_t meta0 = ring_meta[0];
                uint32_t meta1 = ring_meta[1];
                uint32_t meta2 = ring_meta[2];
                uint64_t buffer_scratch_noc_addr = buffer_scratch_noc_addr_table[c][slot];
                read_slots[c] = (slot + 1) & SLOTS_PER_UNTILIZER_MASK;

                if (meta0 == ROUTE_INFO_SENTINEL) {
                    // Reset the sem so a subsequent kernel invocation starts at 0 even if the
                    // framework doesn't reset program-level sems between runs.  Pairs with
                    // consumed[c] being a local that resets at kernel entry.
                    noc_semaphore_set(data_ready_sem_ptrs[c], 0);
                    noc_async_atomic_barrier();
                    untilizer_finished[c] = true;
                    untilizer_done_count++;
                    continue;
                }

                uint32_t dst_chip = meta0;
                uint32_t output_page_idx = meta1 * num_experts_per_tok + meta2;

                if constexpr (is_1d_topology<topology>()) {
                    uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);
                    uint32_t distance =
                        manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);

                    cb_reserve_back(cb_route_info_id, 1);
                    uint32_t cb_base = get_write_ptr(cb_route_info_id);
                    volatile tt_l1_ptr uint32_t* route_info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base);
                    route_info[0] = route;
                    route_info[1] = distance;
                    route_info[2] = output_page_idx;
                    // FABRIC_2D: writer recomputes the EDM direction from route_info[3] (dst_chip)
                    // and ignores slots [0..1]. All four slots are written unconditionally so the
                    // 2D writer doesn't see uninitialized garbage in the dst_chip slot.
                    route_info[3] = dst_chip;
                    {
                        // DeviceZoneScopedN("sending-for-FABRIC-write");
                        uint32_t output_dst = cb_base + l1_alignment;
                        noc_async_read(buffer_scratch_noc_addr, output_dst, aligned_output_page_size);
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_route_info_id, 1);
                }
                noc_semaphore_inc<true>(untilizer_credits_noc_addrs[c], 1);
            }
        }
#endif
    }

    // Push sentinel to signal writer that all dispatches are done
    cb_reserve_back(cb_route_info_id, 1);
    volatile tt_l1_ptr uint32_t* route_info =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
    route_info[0] = ROUTE_INFO_SENTINEL;
    route_info[1] = 0;
    route_info[2] = 0;
    route_info[3] = 0;
#if USE_STORE_AND_FORWARD
    // The relaying writer dispatches on the command word, so termination is a command rather than a
    // sentinel in the route slot.
    route_info[sf::HDR_CMD] = sf::CMD_DONE;
#endif
    cb_push_back(cb_route_info_id, 1);

#if USE_STORE_AND_FORWARD
    // The cross-chip counters live on GlobalSemaphores, which the framework zeroes when it creates
    // them and never again -- so without this, a second invocation would open with an arrival count
    // already past the end-of-stream bias and read pages that were never written.
    //
    // Waiting for the writer's exit barrier is what makes this safe: it proves every peer has issued
    // its last data and end-of-stream increments, and no peer can begin the next invocation's sends
    // before that invocation's init barrier, which it cannot reach until we finish here.  Subtract
    // rather than assign, so an increment that arrives against expectation shows up as a residue the
    // magic-word and distance checks will catch, instead of being silently dropped.
    {
        Semaphore<> sf_cleanup_sem(sf_cleanup_semaphore_id);
        sf_cleanup_sem.wait(1);
        for (uint32_t d = 0; d < 2; d++) {
            for (uint32_t r = 0; r < sf_levels; r++) {
                for (const uint32_t addr : {sf_arrived_addr[d][r], sf_credit_addr[d][r]}) {
                    invalidate_l1_cache();
                    const uint32_t observed = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
                    if (observed != 0) {
                        noc_semaphore_inc(get_noc_addr(addr), (uint32_t)(-(int32_t)observed));
                    }
                }
            }
        }
        noc_async_atomic_barrier();
        sf_cleanup_sem.set(0);
    }
#endif
}
