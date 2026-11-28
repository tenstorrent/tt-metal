// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch port sender (BRISC, port core (lane, d)): owns the fabric connection to the neighbour in direction
// d. In order:
//   1. fence: raise the neighbour's opposite port's `fence` word, wait for our own, take it back to zero; only then may
//      packets go into the neighbour's landing ring;
//   2. slice exchange: send this chip's plan-table slot o to every owner o whose first hop is this port's direction (on
//      this link in turn), into o's inbox slot `me`, the last packet fused with an increment of o's `inbox` word on
//      its planner 0 (which copies the chip's own slot, waits for the N - 1 others, and releases the planners);
//   3. traversal: packets of up to B entries (forwarded queue entries first, then this chip's own list) into the
//      neighbour's landing slot `sent % slots` with a fused increment of its `arrived` word, END(g) generations as in
//      sparse_dispatch_2d. It returns the credits of this core's unpacker (packets from the same neighbour) to the
//      neighbour's `credit` word in batches of kCreditBatch (at the start of a pass and after each packet it builds),
//      and all of them before it waits or idles: it never waits while owing credits;
//   4. exit when END(GMAX) is out, every input delivered END(GMAX), every slot is credited, and this core's unpacker is
//      done with all its credits returned; it then takes `credit` back to zero.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#ifdef KVD_MUX
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#endif
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#include "kvd_common.hpp"

#include <cstdint>

using namespace kvd;
namespace fabric_mesh = tt::tt_fabric::mesh::experimental;

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t lanes = get_compile_time_arg_val(1);  // links x lanes per link
    constexpr uint32_t stripe = get_compile_time_arg_val(2);
    constexpr uint32_t owner_slot_rows = get_compile_time_arg_val(3);
    constexpr uint32_t inbox_slot_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t B = get_compile_time_arg_val(6);
    constexpr uint32_t slots = get_compile_time_arg_val(7);
    constexpr uint32_t gmax = get_compile_time_arg_val(8);
    constexpr uint32_t planners = get_compile_time_arg_val(9);
    constexpr uint32_t pages_own = get_compile_time_arg_val(10);
    constexpr uint32_t pages_per_slot = get_compile_time_arg_val(11);
    constexpr bool has_batch = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t max_payload = get_compile_time_arg_val(13) & ~63u;  // 64 B multiples keep DRAM reads aligned
    constexpr uint32_t cb_land = get_compile_time_arg_val(14);
    constexpr uint32_t cb_send = get_compile_time_arg_val(15);
    constexpr uint32_t cb_qin = get_compile_time_arg_val(16);
    constexpr uint32_t cb_misc = get_compile_time_arg_val(17);
    constexpr uint32_t num_slots_ct = get_compile_time_arg_val(18);  // cache slots
    constexpr uint32_t owner_blocks_max = get_compile_time_arg_val(19);
#ifdef KVD_MUX
    // the mux connection's compile-time args (ttnn::ccl::fabric_mux_connection_ct_args), then the accessors
    constexpr uint8_t mux_num_buffers = get_compile_time_arg_val(20);
    constexpr size_t mux_buffer_bytes = get_compile_time_arg_val(21);
    constexpr uint32_t mux_l1_base_semaphore_id = get_compile_time_arg_val(22);
    constexpr size_t mux_termination_offset = get_compile_time_arg_val(23);
    constexpr uint32_t lanes_per_link = get_compile_time_arg_val(24);
    constexpr uint32_t acc0 = 25;
#else
    constexpr uint32_t acc0 = 20;
#endif
    constexpr auto input_args = TensorAccessorArgs<acc0>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto inbox_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto scratch_args = TensorAccessorArgs<inbox_args.next_compile_time_args_offset()>();
    constexpr auto batch_args = TensorAccessorArgs<scratch_args.next_compile_time_args_offset()>();
    constexpr uint32_t sbytes = slot_bytes(B, row_bytes);
    constexpr uint32_t rows_off = slot_rows_offset(B);
    constexpr uint32_t words_per_block = stripe / 32;
    constexpr uint32_t bitmap_off = ::sparse_push::plan_table::kBitmapOff;
    constexpr uint32_t dest_chip_ids[N] = DEST_CHIP_ID;
    constexpr uint32_t dest_mesh_ids[N] = DEST_MESH_ID;

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t inbox_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t scratch_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t batch_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(5);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(6);
    const uint32_t host_slot = get_common_arg_val<uint32_t>(7);
    const uint32_t fence_addr = get_common_arg_val<uint32_t>(8);
    const uint32_t arrived_addr = get_common_arg_val<uint32_t>(9);
    const uint32_t credit_addr = get_common_arg_val<uint32_t>(10);
    const uint32_t inbox_sem_addr = get_common_arg_val<uint32_t>(11);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(12);

    uint32_t a = 0;
    const uint32_t present = get_arg_val<uint32_t>(a++);
    if (!present) {
        return;
    }
    const uint32_t d = get_arg_val<uint32_t>(a++);
    const uint32_t lane = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t eth_dir = get_arg_val<uint32_t>(a++);  // this port's fabric direction
    const uint32_t nb_chip = get_arg_val<uint32_t>(a++);
    const uint32_t nb_mesh = get_arg_val<uint32_t>(a++);
    const uint32_t nb_x = get_arg_val<uint32_t>(a++);  // the neighbour's port core facing back (same on every chip)
    const uint32_t nb_y = get_arg_val<uint32_t>(a++);
    const uint32_t sync_x = get_arg_val<uint32_t>(a++);  // planner 0: owns the inbox word (same core on every chip)
    const uint32_t sync_y = get_arg_val<uint32_t>(a++);
    const uint32_t own_ready_sem = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t unpack_done_sem = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t credit_out_sem = get_semaphore(get_arg_val<uint32_t>(a++));
    // the three input streams: on, base scratch page, tail semaphore id
    uint32_t q_on[3], q_base[3], q_tail_sem[3];
    for (uint32_t k = 0; k < 3; ++k) {
        q_on[k] = get_arg_val<uint32_t>(a++);
        q_base[k] = get_arg_val<uint32_t>(a++);
        q_tail_sem[k] = get_semaphore(get_arg_val<uint32_t>(a++));
    }
    const uint32_t own_base = get_arg_val<uint32_t>(a++);  // scratch page of (this port, planner 0)'s list
    const uint32_t count_page = get_arg_val<uint32_t>(a++);
    size_t fabric_arg = a;

    const auto input = TensorAccessor(input_args, input_addr, row_bytes);
    const auto output = TensorAccessor(output_args, output_addr, row_bytes);
    const auto table = TensorAccessor(table_args, table_addr, inbox_slot_bytes);
    const auto inbox = TensorAccessor(inbox_args, inbox_addr, inbox_slot_bytes);
    const auto scratch = TensorAccessor(scratch_args, scratch_addr, kScratchPageBytes);

    cb_reserve_back(cb_misc, 1);
    const uint32_t misc_l1 = align64(get_write_ptr(cb_misc));
    uint32_t slot = host_slot;
    if constexpr (has_batch) {
        const auto batch = TensorAccessor(batch_args, batch_addr);
        noc_async_read(batch.get_noc_addr(0), misc_l1, sizeof(uint32_t));
        noc_async_read_barrier();
        const uint32_t user = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(misc_l1);
        slot = user < num_slots_ct / num_layers ? user * num_layers + layer_idx : 0;  // never read past the cache
    }
    const uint32_t slot_base = slot * pages_per_slot;

#ifdef KVD_MUX
    // A program-local mux shares this (direction, link) between its lanes: the connection, in the order
    // ttnn::ccl::fabric_mux_connection_rt_args lays it out.
    size_t ma = fabric_arg + 1;  // (mux_connection_valid: always set)
    const bool is_termination_master = get_arg_val<uint32_t>(ma++) != 0;
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(ma++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(ma++);
    const size_t fabric_mux_channel_base_offset = get_arg_val<uint32_t>(ma++);
    const size_t fabric_mux_connection_info_offset = get_arg_val<uint32_t>(ma++);
    const size_t fabric_mux_connection_handshake_offset = get_arg_val<uint32_t>(ma++);
    const size_t fabric_mux_flow_control_offset = get_arg_val<uint32_t>(ma++);
    const size_t fabric_mux_buffer_index_offset = get_arg_val<uint32_t>(ma++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(ma++);
    const uint32_t termination_sync_id = get_arg_val<uint32_t>(ma++);
    const uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(ma++));
    const uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(ma++));
    const uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(ma++));
    const uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(ma++));
    const uint8_t termination_master_noc_x = get_arg_val<uint32_t>(ma++);
    const uint8_t termination_master_noc_y = get_arg_val<uint32_t>(ma++);
    uint32_t mux_l1_base = 0;
    auto conn = tt::tt_fabric::build_connection_to_program_local_fabric_mux<mux_num_buffers>(
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_id,
        mux_num_buffers,
        mux_buffer_bytes,
        mux_l1_base_semaphore_id,
        fabric_mux_channel_base_offset,
        fabric_mux_connection_info_offset,
        fabric_mux_connection_handshake_offset,
        fabric_mux_flow_control_offset,
        fabric_mux_buffer_index_offset,
        local_flow_control_address,
        local_teardown_address,
        local_buffer_index_address,
        local_fabric_mux_status_address,
        mux_l1_base);
    tt::tt_fabric::fabric_client_connect(conn);
#else
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fabric_arg);
    conn.open();
#endif
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[kRoundPackets];
    for (uint32_t i = 0; i < kRoundPackets; ++i) {
        hdrs[i] = PacketHeaderPool::allocate_header();
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* chdr = PacketHeaderPool::allocate_header();

    const uint32_t t_k0 = stats_clock();
    // ---- 1. fence with the neighbour
    {
        KVD_ZONE("KVD-S-fence");
        fabric_mesh::fabric_unicast_noc_unicast_atomic_inc(
            &conn,
            chdr,
            nb_chip,
            nb_mesh,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{get_noc_addr(nb_x, nb_y, fence_addr), 1, true});
        noc_async_writes_flushed();
        auto* fence = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fence_addr);
        noc_semaphore_wait_min(fence, 1);
        noc_semaphore_inc(get_noc_addr(fence_addr), 0u - 1u);
        noc_async_atomic_barrier();
    }

    const uint32_t t_fence = stats_clock();
    // ---- 2. slice exchange: this chip's plan-table slot o -> owner o's inbox slot `me`
    cb_reserve_back(cb_send, 1);
    const uint32_t send_l1 = align64(get_write_ptr(cb_send));
    noc_async_read(table.get_noc_addr(0), misc_l1, 64);
    noc_async_read_barrier();
    const uint32_t active_blocks =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(misc_l1)[::sparse_push::plan_table::kWordActiveBlocks];
    auto owned_by = [&](uint32_t o) {
        const uint32_t b = active_blocks > o ? (active_blocks - o + N - 1) / N : 0u;
        return b < owner_blocks_max ? b : owner_blocks_max;  // a header never overruns a slot
    };
    uint32_t nth = 0;
    {
        KVD_ZONE("KVD-S-xchg");
        for (uint32_t o = 0; o < N; ++o) {
            if (o == me) {
                continue;
            }
            if (static_cast<uint32_t>(
                    tt::tt_fabric::get_next_hop_router_direction(dest_mesh_ids[o], dest_chip_ids[o])) != eth_dir) {
                continue;
            }
            if (nth++ % lanes != lane) {
                continue;
            }
            const uint32_t bytes = ((bitmap_off + owned_by(o) * words_per_block) * 4 + 63) & ~63u;
            const uint64_t inbox_dst = tt::tt_fabric::linear::addrgen_detail::get_noc_address(inbox, me);
            const uint64_t remote_inbox_sem = get_noc_addr(sync_x, sync_y, inbox_sem_addr);
            for (uint32_t off = 0; off < bytes;) {
                const uint32_t n = bytes - off < max_payload ? bytes - off : max_payload;
                noc_async_read(table.get_noc_addr(o) + off, send_l1, n);
                noc_async_read_barrier();
                if (off == 0) {
                    // send this call's extent (slot 0's header)
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        send_l1)[::sparse_push::plan_table::kWordActiveBlocks] = active_blocks;
                }
                if (off + n == bytes) {
                    fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                        &conn,
                        hdr,
                        dest_chip_ids[o],
                        dest_mesh_ids[o],
                        send_l1,
                        n,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            inbox_dst + off, remote_inbox_sem, 1, true});
                } else {
                    fabric_mesh::fabric_unicast_noc_unicast_write(
                        &conn,
                        hdr,
                        dest_chip_ids[o],
                        dest_mesh_ids[o],
                        send_l1,
                        n,
                        tt::tt_fabric::NocUnicastCommandHeader{inbox_dst + off});
                }
                noc_async_writes_flushed();
                off += n;
            }
        }
    }
    const uint32_t t_xchg = stats_clock();
    // ---- own lists: every planner's length for this port (returning credits while waiting: this core's unpacker may
    // already take the neighbour's packets)
    auto* credit_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_out_sem);
    uint32_t returned = 0;
    const uint64_t nb_credit = get_noc_addr(nb_x, nb_y, credit_addr);
    // Credits go back in batches (a fabric packet and a flush each): when at least kCreditBatch are owed, and all of
    // them whenever this sender would otherwise wait or idle.
    constexpr uint32_t kCreditBatch = slots < 4 ? slots : 4;
    auto return_credits = [&](uint32_t at_least = 1) {
        const uint32_t owed = *credit_out - returned;
        if (owed >= at_least) {
            fabric_mesh::fabric_unicast_noc_unicast_atomic_inc(
                &conn, chdr, nb_chip, nb_mesh, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{nb_credit, owed, true});
            noc_async_writes_flushed();
            returned += owed;
        }
    };

    auto* own_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own_ready_sem);
    {
        KVD_ZONE("KVD-S-ownwait");
        while (true) {
            if (*own_ready >= planners) {
                break;
            }
            return_credits();
        }
    }
    noc_semaphore_set(own_ready, 0);
    for (uint32_t p = 0; p < planners; ++p) {
        noc_async_read(scratch.get_noc_addr(count_page) + p * 64, misc_l1 + 64 + p * 64, 64);
    }
    noc_async_read_barrier();
    auto own_count = [&](uint32_t p) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(misc_l1 + 64 + p * 64)[0];
    };

    const uint32_t t_own = stats_clock();
    // ---- 3. traversal
    cb_reserve_back(cb_qin, 1);
    const uint32_t qin_l1 = align64(get_write_ptr(cb_qin));  // 4 read buffers: 3 streams + own list, one page each
    auto* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_addr);
    auto* unpack_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(unpack_done_sem);
    volatile tt_l1_ptr uint32_t* tail[3];
    for (uint32_t k = 0; k < 3; ++k) {
        tail[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(q_tail_sem[k]);
    }
    uint32_t q_read[3] = {0, 0, 0};
    int32_t end_in[3] = {-1, -1, -1};
    int32_t end_out = -1;
    uint32_t own_p = 0, own_k = 0, own_buf_page = ~0u;
    uint32_t sent = 0;
    const uint64_t nb_land = get_noc_addr(nb_x, nb_y, align64(get_write_ptr(cb_land)));
    const uint64_t nb_arrived = get_noc_addr(nb_x, nb_y, arrived_addr);

    // Input streams: each has an L1 ring of one DRAM page (kEntriesPerPage entries, at the same offsets as in its
    // page). Entries the producer published are fetched asynchronously (q_fetched), and the round's read barrier
    // completes them, so the next round finds them in L1 without a blocking read. q_ready: entries known complete.
    uint32_t q_fetched[3] = {0, 0, 0}, q_ready[3] = {0, 0, 0};
    auto q_prefetch = [&](uint32_t k) {
        if (!q_on[k]) {
            return;
        }
        const uint32_t tail_seen = *tail[k];
        // never run more than a ring (less one 64 B line: a read rounds up to it) ahead of what is consumed, so a fetch
        // cannot overwrite an unconsumed entry of the previous page
        const uint32_t cap = q_read[k] + kEntriesPerPage - 8;
        const uint32_t limit = tail_seen < cap ? tail_seen : cap;
        const uint32_t buf = qin_l1 + k * kScratchPageBytes;
        while (q_fetched[k] < limit) {
            const uint32_t page = q_fetched[k] / kEntriesPerPage;
            const uint32_t page_end = (page + 1) * kEntriesPerPage;
            const uint32_t hi = limit < page_end ? limit : page_end;
            const uint32_t from = ((q_fetched[k] % kEntriesPerPage) * kEntryBytes) & ~63u;
            const uint32_t to = (((hi - 1) % kEntriesPerPage) * kEntryBytes + kEntryBytes + 63) & ~63u;
            noc_async_read(scratch.get_noc_addr(q_base[k] + page) + from, buf + from, to - from);
            q_fetched[k] = hi;
        }
    };
    auto q_entry = [&](uint32_t k, uint32_t i) -> volatile tt_l1_ptr uint32_t* {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            qin_l1 + k * kScratchPageBytes + (i % kEntriesPerPage) * kEntryBytes);
    };
    auto own_entry = [&]() -> volatile tt_l1_ptr uint32_t* {
        const uint32_t page = own_k / kEntriesPerPage;
        const uint32_t buf = qin_l1 + 3 * kScratchPageBytes;
        const uint32_t key = own_p * pages_own + page;
        if (own_buf_page != key) {
            noc_async_read(scratch.get_noc_addr(own_base + key), buf, kScratchPageBytes);
            noc_async_read_barrier();
            own_buf_page = key;
        }
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + (own_k % kEntriesPerPage) * kEntryBytes);
    };
    bool send_done = false;
    KVD_ZONE("KVD-S-traverse");
    auto now = stats_clock;
    uint32_t cy_credit = 0, cy_build = 0, cy_idle = 0, cy_send = 0, n_rounds = 0, n_packets = 0, n_rows = 0;
    uint32_t cy_rd = 0, cy_own = 0, n_q = 0, n_own = 0;
    const uint32_t t_start = now();
#ifdef KVD_ABL_SYNTH
    // Transport ceiling: KVD_ABL_SYNTH rows straight from L1 (garbage) to the neighbour, KVD_ABL_CHUNKS a packet
    // (0: today's B-row packet into the landing ring; 1: one row a packet into a gather-buffer page in DRAM; 2-4:
    // a scatter write of that many rows into DRAM pages), then one atomic inc the neighbour's unpacker waits for.
    {
#ifdef KVD_ABL_CHUNKS
        constexpr uint32_t chunks = KVD_ABL_CHUNKS;
#else
        constexpr uint32_t chunks = 4;
#endif
        constexpr uint32_t per_pkt = (chunks == 0 || chunks >= 5) ? B : chunks;
        const uint32_t total_pages = owner_slot_rows * N;
        uint32_t page = me * 7919u % total_pages;
        auto next_page = [&]() {
            page = (page + 104729u) % total_pages;
            return tt::tt_fabric::linear::addrgen_detail::get_noc_address(output, page);
        };
        uint32_t pk = 0;
#ifdef KVD_ABL_SYNTH_STATE
        // headers built once (route, payload size); per packet only the destination address changes
        for (uint32_t k = 0; k < kRoundPackets; ++k) {
            fabric_mesh::fabric_unicast_noc_unicast_write_set_state<
                tt::tt_fabric::common::experimental::UnicastWriteUpdateMask::None>(
                hdrs[k], nb_chip, nb_mesh, tt::tt_fabric::NocUnicastCommandHeader{nb_land}, rows_off + B * row_bytes);
        }
#endif
        uint32_t cy_slot = 0;
        const uint32_t ty0 = stats_clock();
        for (uint32_t i = 0; i < KVD_ABL_SYNTH; i += per_pkt, ++pk) {
#ifdef KVD_STATS
            {
                const uint32_t tw = stats_clock();
                conn.wait_for_empty_write_slot();
                cy_slot += stats_clock() - tw;
            }
#endif
#ifndef KVD_ABL_SYNTH_NOFLUSH
            if (pk % kRoundPackets == 0) {
                noc_async_writes_flushed();  // the headers are reused
            }
#endif
            auto* h = hdrs[pk % kRoundPackets];
            if constexpr (chunks == 6) {  // fused, flush = false
                const uint64_t dst = nb_land + (pk % slots) * sbytes;
                fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                    &conn,
                    h,
                    nb_chip,
                    nb_mesh,
                    send_l1,
                    rows_off + B * row_bytes,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, dst + 4, 1, false});
                continue;
            }
            if constexpr (chunks == 7) {  // plain write; every 4th packet also a header-only atomic inc (batched
                                          // arrivals)
                const uint64_t dst = nb_land + (pk % slots) * sbytes;
                fabric_mesh::fabric_unicast_noc_unicast_write(
                    &conn,
                    h,
                    nb_chip,
                    nb_mesh,
                    send_l1,
                    rows_off + B * row_bytes,
                    tt::tt_fabric::NocUnicastCommandHeader{dst});
                if ((pk & 3) == 3) {
                    noc_async_writes_flushed();  // (chdr reuse)
                    fabric_mesh::fabric_unicast_noc_unicast_atomic_inc(
                        &conn,
                        chdr,
                        nb_chip,
                        nb_mesh,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dst + 4, 4, true});
                }
                continue;
            }
            if constexpr (chunks == 5) {  // fused write + atomic inc (v3's packet), the inc on a dummy word in the slot
                const uint64_t dst = nb_land + (pk % slots) * sbytes;
                fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                    &conn,
                    h,
                    nb_chip,
                    nb_mesh,
                    send_l1,
                    rows_off + B * row_bytes,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, dst + 4, 1, true});
                continue;
            }
            if constexpr (chunks == 0) {
#ifdef KVD_ABL_SYNTH_STATE
                fabric_mesh::fabric_unicast_noc_unicast_write_with_state<
                    tt::tt_fabric::common::experimental::UnicastWriteUpdateMask::DstAddr>(
                    &conn, h, send_l1, tt::tt_fabric::NocUnicastCommandHeader{nb_land + (pk % slots) * sbytes});
                continue;
#endif
                fabric_mesh::fabric_unicast_noc_unicast_write(
                    &conn,
                    h,
                    nb_chip,
                    nb_mesh,
                    send_l1,
                    rows_off + B * row_bytes,
                    tt::tt_fabric::NocUnicastCommandHeader{nb_land + (pk % slots) * sbytes});
            } else if constexpr (chunks == 1) {
                fabric_mesh::fabric_unicast_noc_unicast_write(
                    &conn,
                    h,
                    nb_chip,
                    nb_mesh,
                    send_l1,
                    row_bytes,
                    tt::tt_fabric::NocUnicastCommandHeader{next_page()});
            } else {
                uint64_t addrs[chunks];
                uint16_t sizes[chunks];
                for (uint32_t c = 0; c < chunks; ++c) {
                    addrs[c] = next_page();
                    sizes[c] = row_bytes;
                }
                fabric_mesh::fabric_unicast_noc_scatter_write(
                    &conn,
                    h,
                    nb_chip,
                    nb_mesh,
                    send_l1,
                    chunks * row_bytes,
                    tt::tt_fabric::NocUnicastScatterCommandHeader(addrs, sizes, chunks));
            }
        }
        noc_async_writes_flushed();
#ifdef KVD_STATS
        DPRINT("KVDY d={} l={} total={} slotwait={} pkts={}\n", d, lane, stats_clock() - ty0, cy_slot, pk);
#endif
        (void)ty0;
        (void)cy_slot;
        fabric_mesh::fabric_unicast_noc_unicast_atomic_inc(
            &conn, chdr, nb_chip, nb_mesh, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{nb_arrived, 1, true});
        noc_async_writes_flushed();
    }
#endif
    while (true) {
#if defined(KVD_ABL_NO_TRAVERSE) || defined(KVD_ABL_SYNTH)
        break;  // fence, exchange and planners only (and the synthetic transport)
#endif
        uint32_t t0 = now();
        return_credits(kCreditBatch);
        if (send_done) {
            return_credits();
            if (*unpack_done != 0 && *credit_out == returned) {
                break;
            }
            continue;
        }
        if (sent - *credit >= slots) {
            return_credits();
            cy_credit += now() - t0;
            continue;  // no free landing slot on the neighbour: keep returning credits
        }
        // A round: up to kRoundPackets packets (as many as there are free landing slots), every row read issued, one
        // read barrier, then all of them sent (one header each) and one flush: the DRAM read latency is paid once a
        // round, not once a packet.
        const uint32_t free_slots = slots - (sent - *credit);
        const uint32_t max_packets = free_slots < kRoundPackets ? free_slots : kRoundPackets;
        uint32_t pkt_n[kRoundPackets], pkt_rows[kRoundPackets];
        uint32_t packets = 0;
        while (packets < max_packets) {
            const uint32_t slot_l1 = send_l1 + packets * sbytes;
            auto* ents = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1 + kSlotHeaderBytes);
            uint32_t n = 0, rows = 0;
            while (n < B) {
                bool took = false;
                for (uint32_t k = 0; k < 3 && n < B; ++k) {
                    if (!q_on[k]) {
                        continue;
                    }
                    if (q_read[k] >= q_ready[k]) {
                        continue;
                    }
                    auto* e = q_entry(k, q_read[k]);
                    const uint32_t w0 = e[0], w1 = e[1];
                    ++q_read[k];
                    took = true;
                    if (w0 & kEndBit) {
                        end_in[k] = static_cast<int32_t>(entry_gen(w0));
                        continue;
                    }
                    const uint32_t g = entry_row(w0);
                    const uint32_t owner = (g / stripe) % N;
                    const uint32_t tr = now();
#ifndef KVD_ABL_NO_ROW_READ
                    noc_async_read(
                        output.get_noc_addr(owner * owner_slot_rows + (g / stripe / N) * stripe + g % stripe),
                        slot_l1 + rows_off + rows * row_bytes,
                        row_bytes);
#else
                    (void)owner;
#endif
                    cy_rd += now() - tr;
                    ents[2 * n] = w0;
                    ents[2 * n + 1] = w1;
                    ++n;
                    ++rows;
                    ++n_q;
                }
                if (took) {
                    continue;
                }
                // own rows when no input entry is waiting
                while (own_p < planners && own_k >= own_count(own_p)) {
                    ++own_p;
                    own_k = 0;
                }
                if (own_p < planners) {
                    const uint32_t to = now();
                    auto* e = own_entry();
                    const uint32_t w0 = e[0], w1 = e[1];
                    ++own_k;
                    const uint32_t g = entry_row(w0);
                    const uint32_t local_row = (g / stripe / N) * stripe + g % stripe;
                    const uint32_t tr = now();
#ifndef KVD_ABL_NO_ROW_READ
                    noc_async_read(
                        input.get_noc_addr(slot_base + local_row), slot_l1 + rows_off + rows * row_bytes, row_bytes);
#else
                    (void)local_row;
#endif
                    cy_rd += now() - tr;
                    cy_own += now() - to;
                    ++n_own;
                    ents[2 * n] = w0;
                    ents[2 * n + 1] = w1;
                    ++n;
                    ++rows;
                    continue;
                }
                // END(g) once own rows are out and every input delivered END(g - 1)
                if (end_out < static_cast<int32_t>(gmax)) {
                    const int32_t g = end_out + 1;
                    bool ok = true;
                    for (uint32_t k = 0; k < 3; ++k) {
                        if (q_on[k] && end_in[k] < g - 1) {
                            ok = false;
                        }
                    }
                    if (g == 0 || ok) {
                        ents[2 * n] = end_word0(static_cast<uint32_t>(g));
                        ents[2 * n + 1] = 0;
                        ++n;
                        end_out = g;
                        continue;
                    }
                }
                break;
            }
            if (n == 0) {
                break;
            }
            pkt_n[packets] = n;
            pkt_rows[packets] = rows;
            ++packets;
            return_credits(kCreditBatch);
            if (n < B) {
                break;  // nothing more is ready: send what this round has
            }
        }
        const uint32_t t1 = now();
        if (packets == 0) {
            return_credits();
            // nothing ready in L1: fetch what the producers published since, and wait for it
            for (uint32_t k = 0; k < 3; ++k) {
                q_prefetch(k);
            }
            noc_async_read_barrier();
            for (uint32_t k = 0; k < 3; ++k) {
                q_ready[k] = q_fetched[k];
            }
            cy_idle += t1 - t0;
            bool inputs_ended = true;
            for (uint32_t k = 0; k < 3; ++k) {
                if (q_on[k] && end_in[k] < static_cast<int32_t>(gmax)) {
                    inputs_ended = false;
                }
            }
            if (end_out == static_cast<int32_t>(gmax) && inputs_ended && *credit == sent) {
                send_done = true;
            }
            continue;
        }
        for (uint32_t k = 0; k < 3; ++k) {
            q_prefetch(k);  // completed by this round's barrier, used next round
        }
        {
            KVD_ZONE("KVD-S-rdbar");
            noc_async_read_barrier();
        }
        for (uint32_t k = 0; k < 3; ++k) {
            q_ready[k] = q_fetched[k];
        }
        KVD_ZONE("KVD-S-send");
        for (uint32_t i = 0; i < packets; ++i) {
            const uint32_t slot_l1 = send_l1 + i * sbytes;
            auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1);
            h[0] = pkt_n[i];
            h[1] = pkt_rows[i];
            fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                &conn,
                hdrs[i],
                nb_chip,
                nb_mesh,
                slot_l1,
#ifdef KVD_ABL_HDR_ONLY
                rows_off,  // entries only: no rows on the wire
#else
                rows_off + pkt_rows[i] * row_bytes,
#endif
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    nb_land + (sent % slots) * sbytes, nb_arrived, 1, true});
            ++sent;
        }
        noc_async_writes_flushed();  // the send buffers and headers are reused next round
        cy_build += t1 - t0;
        cy_send += now() - t1;
        ++n_rounds;
        n_packets += packets;
        for (uint32_t i = 0; i < packets; ++i) {
            n_rows += pkt_rows[i];
        }
    }
#ifdef KVD_STATS
    DPRINT(
        "KVDS d={} l={} fence={} xchg={} ownwait={} total={} credit={} build={} idle={} send={} rounds={} pkts={} "
        "rows={} rd={} own={} nq={} nown={}\n",
        d,
        lane,
        t_fence - t_k0,
        t_xchg - t_fence,
        t_own - t_xchg,
        now() - t_start,
        cy_credit,
        cy_build,
        cy_idle,
        cy_send,
        n_rounds,
        n_packets,
        n_rows,
        cy_rd,
        cy_own,
        n_q,
        n_own);
#endif
    // `credit` is persistent: leave it at zero for the next call.
    noc_semaphore_inc(get_noc_addr(credit_addr), 0u - sent);
    noc_async_atomic_barrier();
#ifdef KVD_MUX
    // The mux forwards asynchronously: let every slot drain, disconnect, and the termination master shuts the mux down
    // once the link's other lanes have disconnected.
    while (conn.get_num_free_write_slots() != mux_num_buffers) {
    }
    tt::tt_fabric::fabric_client_disconnect(conn);
    auto* term_sync = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(termination_sync_id));
    if (is_termination_master) {
        noc_semaphore_wait_min(term_sync, lanes_per_link - 1);
        noc_semaphore_set(term_sync, 0);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, mux_l1_base + mux_termination_offset);
    } else {
        noc_semaphore_inc(
            get_noc_addr(termination_master_noc_x, termination_master_noc_y, get_semaphore(termination_sync_id)), 1);
        noc_async_atomic_barrier();
    }
#else
    conn.close();
#endif
}
