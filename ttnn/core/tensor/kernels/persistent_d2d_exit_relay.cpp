// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Exit relay for D2DStreamService relay mode (D2DStreamConfig::relay).
//
// Runs on a service core of a sender-mesh chip that has a cable to the receiver mesh
// (the exit chip). Some sender coords ("sources") are assigned to this cable. Slot k is
// the k-th source. For a remote source, the unchanged persistent_d2d_sender.cpp writes its
// tensor into this chip's staging tensor k and its metadata into md slot k, then
// Flush-increments arrived[k] once per sender lane. For this chip's own coord (own_slot),
// the workers increment data_ready directly, as they do for a normal sender service.
//
// Per lease window (link_grant == 1), every source is forwarded once, in arrival order:
//   1. source k is ready: its data arrived, and the entry relay has freed slot k on the
//      far side (cable_acked[k] caught up with cable_sent[k]);
//   2. copy the tensor over the cable (Z) into the entry relay's destination for slot k,
//      then the metadata, then Flush-increment the entry relay's arrived[k];
//   3. give the source its credit: bytes_acked +1 on the source's sender service core
//      (over fabric), or consumed_sem multicast to this chip's own workers.
// Then close all connections and write link_grant = 0.
//
// Two lanes, like the sender: the master (RISCV_0 / NOC_0) owns the window and the
// choices; the sub (RISCV_1 / NOC_1) copies the second half of each tensor over the
// second cable link. The master tells the sub which slot to copy through go_slot and
// go_count; the sub answers with done_count. With one cable link (z_lanes == 1) only the
// master runs.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/core/tensor/kernels/d2d_relay_common.hpp"

constexpr uint32_t terminate_value = get_named_compile_time_arg_val("terminate_value");
constexpr uint32_t termination_addr = get_named_compile_time_arg_val("termination_addr");
constexpr uint32_t link_grant_addr = get_named_compile_time_arg_val("link_grant_addr");
constexpr uint32_t is_master = get_named_compile_time_arg_val("is_master");
constexpr uint32_t z_lanes = get_named_compile_time_arg_val("z_lanes");
constexpr uint32_t num_slots = get_named_compile_time_arg_val("num_slots");
// own_slot == num_slots when this chip's coord does not take part.
constexpr uint32_t own_slot = get_named_compile_time_arg_val("own_slot");
constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
constexpr uint32_t max_payload = get_named_compile_time_arg_val("max_payload");
constexpr uint32_t page_start = get_named_compile_time_arg_val("page_start");  // this lane's pages
constexpr uint32_t page_end = get_named_compile_time_arg_val("page_end");
constexpr uint32_t ring_slots = get_named_compile_time_arg_val("ring_slots");
constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
constexpr uint32_t header_cb = get_named_compile_time_arg_val("header_cb");
constexpr uint32_t metadata_enabled = get_named_compile_time_arg_val("metadata_enabled");
constexpr uint32_t metadata_span = get_named_compile_time_arg_val("metadata_span");
// This chip's own workers write their metadata here (the normal sender metadata buffer).
constexpr uint32_t own_md_addr = get_named_compile_time_arg_val("own_md_addr");
// Per-slot words on this core, word_stride apart; md slots are metadata_span apart.
constexpr uint32_t arrived_base = get_named_compile_time_arg_val("arrived_base");
constexpr uint32_t md_base = get_named_compile_time_arg_val("md_base");
constexpr uint32_t cable_acked_base = get_named_compile_time_arg_val("cable_acked_base");
constexpr uint32_t word_stride = get_named_compile_time_arg_val("word_stride");
// Lane sync words (shared L1 of this core).
constexpr uint32_t go_count_addr = get_named_compile_time_arg_val("go_count_addr");
constexpr uint32_t go_slot_addr = get_named_compile_time_arg_val("go_slot_addr");
constexpr uint32_t done_count_addr = get_named_compile_time_arg_val("done_count_addr");
// This chip's own workers (own_slot only).
constexpr uint32_t data_ready_counter_addr = get_named_compile_time_arg_val("data_ready_counter_addr");
constexpr uint32_t consumed_sem_addr = get_named_compile_time_arg_val("consumed_sem_addr");
constexpr uint32_t mcast_x_start = get_named_compile_time_arg_val("mcast_x_start");
constexpr uint32_t mcast_y_start = get_named_compile_time_arg_val("mcast_y_start");
constexpr uint32_t mcast_x_end = get_named_compile_time_arg_val("mcast_x_end");
constexpr uint32_t mcast_y_end = get_named_compile_time_arg_val("mcast_y_end");
constexpr uint32_t num_workers = get_named_compile_time_arg_val("num_workers");
// The entry relay on the other end of the cable.
constexpr uint32_t entry_chip = get_named_compile_time_arg_val("entry_chip");
constexpr uint32_t entry_mesh = get_named_compile_time_arg_val("entry_mesh");
constexpr uint32_t entry_noc_x = get_named_compile_time_arg_val("entry_noc_x");
constexpr uint32_t entry_noc_y = get_named_compile_time_arg_val("entry_noc_y");
constexpr uint32_t entry_arrived_base = get_named_compile_time_arg_val("entry_arrived_base");
constexpr uint32_t entry_md_base = get_named_compile_time_arg_val("entry_md_base");
// Number of in-mesh fabric connections the master holds for credits (one per first-hop
// direction). The sub holds none.
constexpr uint32_t num_credit_conns = get_named_compile_time_arg_val("num_credit_conns");
// Staging, own backing and entry-side tensors all share the per-shard spec.
constexpr auto tensor_args = TensorAccessorArgs<0>();

static_assert(num_slots >= 1 && num_slots <= 32, "pending set is a 32-bit mask");
static_assert(num_credit_conns <= 4, "at most one credit connection per in-mesh direction");

constexpr uint32_t kRingDepth = d2d_relay::floor_pow2(ring_slots);
constexpr uint32_t kTridBase = 1u + (is_master ? 0u : kRingDepth);
static_assert(2u * kRingDepth <= 15u, "two lanes of trids must fit in 0..15");

// Per-slot runtime args. Master: kMasterSlotArgs words per slot; sub: kSubSlotArgs.
//   src_addr, dst_addr                       (both lanes)
//   src_lanes, src_chip, src_mesh, src_noc_x, src_noc_y, src_acked_addr, credit_conn (master)
constexpr uint32_t kSubSlotArgs = 2;
constexpr uint32_t kMasterSlotArgs = 9;

template <typename SlotArgs>
FORCE_INLINE void copy_slot_half(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& z_conn,
    const SlotArgs& slot,
    uint32_t cb_l1_addr,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_header) {
    auto in_acc = TensorAccessor(tensor_args, slot.src_addr);
    auto out_acc = TensorAccessor(tensor_args, slot.dst_addr);
    d2d_relay::stream_pages<page_size, max_payload, kRingDepth, kTridBase>(
        noc, z_conn, in_acc, out_acc, cb_l1_addr, page_start, page_end, data_header);
}

struct Slot {
    uint32_t src_addr;
    uint32_t dst_addr;
};

void kernel_main() {
    size_t arg = 0;
    // Cable (Z) connection on this lane's link. Always the first connection in the args.
    tt::tt_fabric::WorkerToFabricEdmSender z_conn =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    tt::tt_fabric::WorkerToFabricEdmSender credit_conns[num_credit_conns > 0 ? num_credit_conns : 1];
    if constexpr (is_master) {
        for (uint32_t c = 0; c < num_credit_conns; ++c) {
            credit_conns[c] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
        }
    }
    const size_t slot_args = arg;
    auto slot_at = [&](uint32_t k) {
        const size_t base = slot_args + k * (is_master ? kMasterSlotArgs : kSubSlotArgs);
        return Slot{get_arg_val<uint32_t>(base), get_arg_val<uint32_t>(base + 1)};
    };

    // Headers: [0] data to the entry chip, [1] increments to the entry chip,
    // [2] credits (route set per source).
    CircularBuffer hdr_cb(header_cb);
    // Each lane owns 3 headers and Depth ring pages: the two lanes run on two RISCs of one core.
    const uint32_t hdr_l1 = hdr_cb.get_write_ptr() + (is_master ? 0u : 3u * sizeof(PACKET_HEADER_TYPE));
    auto* data_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_l1);
    auto* inc_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_l1 + sizeof(PACKET_HEADER_TYPE));
    auto* credit_header =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_l1 + 2 * sizeof(PACKET_HEADER_TYPE));
    d2d_relay::set_route(data_header, entry_chip, entry_mesh);
    d2d_relay::set_route(inc_header, entry_chip, entry_mesh);

    Noc noc(noc_index);
    CircularBuffer scratch(scratch_cb);
    const uint32_t cb_l1_addr = scratch.get_write_ptr() + (is_master ? 0u : kRingDepth * page_size);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    volatile tt_l1_ptr uint32_t* go_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_count_addr);
    volatile tt_l1_ptr uint32_t* go_slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_slot_addr);
    volatile tt_l1_ptr uint32_t* done_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_count_addr);

    if constexpr (!is_master) {
        // ---- Sub: copy the second half of whichever slot the master names ----
        uint32_t last_go = 0;
        while (true) {
            bool stop = false;
            while (true) {
                if (d2d_relay::terminated(termination_addr, terminate_value)) {
                    stop = true;
                    break;
                }
                if (d2d_relay::advanced(go_count[0], last_go, 1)) {
                    last_go += 1;
                    break;
                }
            }
            if (stop) {
                break;
            }
            const uint32_t k = go_slot[0];
            // Hold the cable link only while copying (the lease may give it to others).
            z_conn.open();
            copy_slot_half(noc, z_conn, slot_at(k), cb_l1_addr, data_header);
            d2d_relay::fabric_atomic_inc(
                z_conn, get_noc_addr(entry_noc_x, entry_noc_y, entry_arrived_base + k * word_stride), true, inc_header);
            z_conn.close();
            done_count[0] += 1;
        }
        return;
    }

    // ---- Master ----
    volatile tt_l1_ptr uint32_t* link_grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(link_grant_addr);
    volatile tt_l1_ptr uint32_t* data_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_counter_addr);
    const uint64_t consumed_mcast =
        get_noc_multicast_addr(mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, consumed_sem_addr);

    // Per-slot progress. last_arrived counts lane increments seen; cable_sent counts
    // chunks sent to the entry relay (compared with cable_acked[k], which it increments).
    uint32_t last_arrived[num_slots] = {};
    uint32_t cable_sent[num_slots] = {};
    uint32_t last_data_ready = 0;
    uint32_t go_local = 0;
    uint32_t last_done = 0;
    constexpr uint32_t kAllSlots = num_slots == 32 ? 0xFFFFFFFFu : ((1u << num_slots) - 1u);

    while (true) {
        // 1. Wait for the lease grant.
        bool stop = false;
        while (true) {
            if (d2d_relay::terminated(termination_addr, terminate_value)) {
                stop = true;
                break;
            }
            if (link_grant[0] == 1) {
                break;
            }
        }
        if (stop) {
            break;
        }
        z_conn.open();
        for (uint32_t c = 0; c < num_credit_conns; ++c) {
            credit_conns[c].open();
        }

        // 2. Forward every source once, in whatever order they become ready.
        uint32_t pending = kAllSlots;
        while (pending != 0 && !stop) {
            for (uint32_t k = 0; k < num_slots; ++k) {
                if ((pending & (1u << k)) == 0) {
                    continue;
                }
                if (d2d_relay::terminated(termination_addr, terminate_value)) {
                    stop = true;
                    break;
                }
                const size_t base = slot_args + k * kMasterSlotArgs;
                const uint32_t src_lanes = get_arg_val<uint32_t>(base + 2);
                volatile tt_l1_ptr uint32_t* arrived =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrived_base + k * word_stride);
                volatile tt_l1_ptr uint32_t* cable_acked =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cable_acked_base + k * word_stride);
                // Far-side slot still busy with the previous chunk.
                if (cable_sent[k] != cable_acked[0]) {
                    continue;
                }
                if (k == own_slot) {
                    if (!d2d_relay::advanced(data_ready[0], last_data_ready, num_workers)) {
                        continue;
                    }
                    last_data_ready += num_workers;
                } else {
                    if (!d2d_relay::advanced(arrived[0], last_arrived[k], src_lanes)) {
                        continue;
                    }
                    last_arrived[k] += src_lanes;
                }

                // Copy: the sub takes the second half on the second link.
                if constexpr (z_lanes > 1) {
                    go_slot[0] = k;
                    go_local += 1;
                    go_count[0] = go_local;
                }
                copy_slot_half(noc, z_conn, slot_at(k), cb_l1_addr, data_header);
                if constexpr (metadata_enabled) {
                    d2d_relay::fabric_write_bytes<max_payload>(
                        z_conn,
                        get_noc_addr(entry_noc_x, entry_noc_y, entry_md_base + k * metadata_span),
                        k == own_slot ? own_md_addr : md_base + k * metadata_span,
                        metadata_span,
                        data_header);
                }
                d2d_relay::fabric_atomic_inc(
                    z_conn,
                    get_noc_addr(entry_noc_x, entry_noc_y, entry_arrived_base + k * word_stride),
                    true,
                    inc_header);
                if constexpr (z_lanes > 1) {
                    while (!d2d_relay::advanced(done_count[0], last_done, 1)) {
                        invalidate_l1_cache();
                    }
                    last_done += 1;
                }
                cable_sent[k] += 1;

                // 3. Credit the source: its staging slot (or the own backing) may be reused.
                if (k == own_slot) {
                    noc_semaphore_inc_multicast(consumed_mcast, 1, num_workers);
                } else {
                    const uint32_t src_chip = get_arg_val<uint32_t>(base + 3);
                    const uint32_t src_mesh = get_arg_val<uint32_t>(base + 4);
                    const uint32_t src_noc_x = get_arg_val<uint32_t>(base + 5);
                    const uint32_t src_noc_y = get_arg_val<uint32_t>(base + 6);
                    const uint32_t src_acked_addr = get_arg_val<uint32_t>(base + 7);
                    const uint32_t credit_conn = get_arg_val<uint32_t>(base + 8);
                    d2d_relay::set_route(credit_header, src_chip, src_mesh);
                    d2d_relay::fabric_atomic_inc(
                        credit_conns[credit_conn],
                        get_noc_addr(src_noc_x, src_noc_y, src_acked_addr),
                        false,
                        credit_header);
                }
                pending &= ~(1u << k);
            }
        }

        // 4. Window done (or terminating): drop every connection, hand the links back.
        noc_async_atomic_barrier();
        z_conn.close();
        for (uint32_t c = 0; c < num_credit_conns; ++c) {
            credit_conns[c].close();
        }
        link_grant[0] = 0;
        if (stop) {
            break;
        }
    }
}
