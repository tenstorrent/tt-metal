// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Entry relay for D2DStreamService relay mode (D2DStreamConfig::relay).
//
// Runs on a service core of the receiver-mesh chip at the far end of a cable (the entry
// chip). Slot k matches slot k of the exit relay: the same coord is source k there and
// target k here. The exit relay writes target k's tensor into this chip's staging tensor k
// (or, for this chip's own coord, straight into its receiver backing tensor), the metadata
// into md slot k, then Flush-increments arrived[k] once per cable lane.
//
// Per lease window (link_grant == 1), every target is served once, in arrival order:
//   * remote target t: when its data arrived and t's receiver has acked the previous chunk
//     (recv_acked[k] == sent[k]), copy staging k into t's backing tensor, the metadata into
//     t's receiver FIFO, Flush-increment t's bytes_sent once per lane (the unchanged
//     persistent_d2d_receiver.cpp waits for that), then credit cable_acked[k] on the exit
//     relay over the cable: staging k is free again.
//   * own target (own_slot): act as this chip's receiver service. Multicast the metadata
//     and data_ready to the workers, then poll (without blocking the other targets) for
//     their consumed acks; credit the exit relay only after that, because the exit relay
//     writes this target straight into the backing tensor the workers read.
// Then close all connections and write link_grant = 0.
//
// Two lanes: the master (RISCV_0 / NOC_0) owns the window; the sub (RISCV_1 / NOC_1)
// copies the second half of a target's tensor on the second link of that direction, when
// that direction has two links (lanes == 2 in the target's args).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/core/tensor/kernels/d2d_relay_common.hpp"

constexpr uint32_t terminate_value = get_named_compile_time_arg_val("terminate_value");
constexpr uint32_t termination_addr = get_named_compile_time_arg_val("termination_addr");
constexpr uint32_t link_grant_addr = get_named_compile_time_arg_val("link_grant_addr");
constexpr uint32_t is_master = get_named_compile_time_arg_val("is_master");
constexpr uint32_t z_lanes = get_named_compile_time_arg_val("z_lanes");  // arrived[k] incs per chunk
constexpr uint32_t num_slots = get_named_compile_time_arg_val("num_slots");
constexpr uint32_t own_slot = get_named_compile_time_arg_val("own_slot");  // num_slots = none
constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
constexpr uint32_t max_payload = get_named_compile_time_arg_val("max_payload");
constexpr uint32_t total_pages = get_named_compile_time_arg_val("total_pages");
constexpr uint32_t half_pages = get_named_compile_time_arg_val("half_pages");  // master [0, half) when 2 lanes
constexpr uint32_t ring_slots = get_named_compile_time_arg_val("ring_slots");
constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
constexpr uint32_t header_cb = get_named_compile_time_arg_val("header_cb");
constexpr uint32_t metadata_enabled = get_named_compile_time_arg_val("metadata_enabled");
constexpr uint32_t metadata_span = get_named_compile_time_arg_val("metadata_span");
constexpr uint32_t metadata_size = get_named_compile_time_arg_val("metadata_size");
constexpr uint32_t worker_md_addr = get_named_compile_time_arg_val("worker_md_addr");
constexpr uint32_t arrived_base = get_named_compile_time_arg_val("arrived_base");
constexpr uint32_t md_base = get_named_compile_time_arg_val("md_base");
constexpr uint32_t recv_acked_base = get_named_compile_time_arg_val("recv_acked_base");
constexpr uint32_t word_stride = get_named_compile_time_arg_val("word_stride");
constexpr uint32_t go_count_addr = get_named_compile_time_arg_val("go_count_addr");
constexpr uint32_t go_slot_addr = get_named_compile_time_arg_val("go_slot_addr");
constexpr uint32_t done_count_addr = get_named_compile_time_arg_val("done_count_addr");
// This chip's own receiver workers (own_slot only).
constexpr uint32_t data_ready_sem_addr = get_named_compile_time_arg_val("data_ready_sem_addr");
constexpr uint32_t consumed_counter_addr = get_named_compile_time_arg_val("consumed_counter_addr");
constexpr uint32_t mcast_x_start = get_named_compile_time_arg_val("mcast_x_start");
constexpr uint32_t mcast_y_start = get_named_compile_time_arg_val("mcast_y_start");
constexpr uint32_t mcast_x_end = get_named_compile_time_arg_val("mcast_x_end");
constexpr uint32_t mcast_y_end = get_named_compile_time_arg_val("mcast_y_end");
constexpr uint32_t num_workers = get_named_compile_time_arg_val("num_workers");
// The exit relay on the other end of the cable.
constexpr uint32_t exit_chip = get_named_compile_time_arg_val("exit_chip");
constexpr uint32_t exit_mesh = get_named_compile_time_arg_val("exit_mesh");
constexpr uint32_t exit_noc_x = get_named_compile_time_arg_val("exit_noc_x");
constexpr uint32_t exit_noc_y = get_named_compile_time_arg_val("exit_noc_y");
constexpr uint32_t exit_cable_acked_base = get_named_compile_time_arg_val("exit_cable_acked_base");
// In-mesh connections this lane holds, one per first-hop direction toward its targets.
constexpr uint32_t num_dir_conns = get_named_compile_time_arg_val("num_dir_conns");
constexpr auto tensor_args = TensorAccessorArgs<0>();

static_assert(num_slots >= 1 && num_slots <= 32, "pending set is a 32-bit mask");
static_assert(num_dir_conns <= 4, "at most one connection per in-mesh direction");

constexpr uint32_t kRingDepth = d2d_relay::floor_pow2(ring_slots);
constexpr uint32_t kTridBase = 1u + (is_master ? 0u : kRingDepth);
static_assert(2u * kRingDepth <= 15u, "two lanes of trids must fit in 0..15");

// Per-slot runtime args (same layout for both lanes):
//   [0] src_addr (staging k)  [1] dst_addr (target backing)  [2] t_chip  [3] t_mesh
//   [4] t_noc_x  [5] t_noc_y  [6] t_fifo_addr  [7] t_bytes_sent_addr  [8] lanes
//   [9] master conn index  [10] sub conn index
constexpr uint32_t kSlotArgs = 11;

struct Target {
    uint32_t src_addr, dst_addr, chip, mesh, noc_x, noc_y, fifo_addr, bytes_sent_addr, lanes, master_conn, sub_conn;
};

FORCE_INLINE Target target_at(size_t slot_args, uint32_t k) {
    const size_t b = slot_args + k * kSlotArgs;
    return Target{
        get_arg_val<uint32_t>(b + 0),
        get_arg_val<uint32_t>(b + 1),
        get_arg_val<uint32_t>(b + 2),
        get_arg_val<uint32_t>(b + 3),
        get_arg_val<uint32_t>(b + 4),
        get_arg_val<uint32_t>(b + 5),
        get_arg_val<uint32_t>(b + 6),
        get_arg_val<uint32_t>(b + 7),
        get_arg_val<uint32_t>(b + 8),
        get_arg_val<uint32_t>(b + 9),
        get_arg_val<uint32_t>(b + 10)};
}

// Copy pages [start, end) of staging k into the target's backing, then Flush-increment the
// target's bytes_sent: the receiver kernel counts one increment per lane.
FORCE_INLINE void copy_to_target(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    const Target& t,
    uint32_t start,
    uint32_t end,
    uint32_t cb_l1_addr,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_header,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* inc_header,
    bool with_metadata,
    uint32_t md_l1_addr) {
    d2d_relay::set_route(data_header, t.chip, t.mesh);
    d2d_relay::set_route(inc_header, t.chip, t.mesh);
    auto in_acc = TensorAccessor(tensor_args, t.src_addr);
    auto out_acc = TensorAccessor(tensor_args, t.dst_addr);
    d2d_relay::stream_pages<page_size, max_payload, kRingDepth, kTridBase>(
        noc, conn, in_acc, out_acc, cb_l1_addr, start, end, data_header);
    if (with_metadata) {
        d2d_relay::fabric_write_bytes<max_payload>(
            conn, get_noc_addr(t.noc_x, t.noc_y, t.fifo_addr), md_l1_addr, metadata_span, data_header);
    }
    d2d_relay::fabric_atomic_inc(conn, get_noc_addr(t.noc_x, t.noc_y, t.bytes_sent_addr), true, inc_header);
}

void kernel_main() {
    size_t arg = 0;
    // Master: cable (Z) connection first, for the credits to the exit relay. Sub: none.
    tt::tt_fabric::WorkerToFabricEdmSender z_conn;
    if constexpr (is_master) {
        z_conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    }
    tt::tt_fabric::WorkerToFabricEdmSender dir_conns[num_dir_conns > 0 ? num_dir_conns : 1];
    for (uint32_t c = 0; c < num_dir_conns; ++c) {
        dir_conns[c] = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    }
    const size_t slot_args = arg;

    // Headers: [0] data, [1] increments to the target, [2] credits to the exit relay.
    CircularBuffer hdr_cb(header_cb);
    // Each lane owns 3 headers and Depth ring pages: the two lanes run on two RISCs of one core.
    const uint32_t hdr_l1 = hdr_cb.get_write_ptr() + (is_master ? 0u : 3u * sizeof(PACKET_HEADER_TYPE));
    auto* data_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_l1);
    auto* inc_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_l1 + sizeof(PACKET_HEADER_TYPE));
    auto* credit_header =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_l1 + 2 * sizeof(PACKET_HEADER_TYPE));
    d2d_relay::set_route(credit_header, exit_chip, exit_mesh);

    Noc noc(noc_index);
    CircularBuffer scratch(scratch_cb);
    const uint32_t cb_l1_addr = scratch.get_write_ptr() + (is_master ? 0u : kRingDepth * page_size);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    volatile tt_l1_ptr uint32_t* go_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_count_addr);
    volatile tt_l1_ptr uint32_t* go_slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_slot_addr);
    volatile tt_l1_ptr uint32_t* done_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_count_addr);

    if constexpr (!is_master) {
        // ---- Sub: copy pages [half, total) of whichever target the master names ----
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
            const Target t = target_at(slot_args, go_slot[0]);
            auto& conn = dir_conns[t.sub_conn];
            conn.open();
            copy_to_target(noc, conn, t, half_pages, total_pages, cb_l1_addr, data_header, inc_header, false, 0);
            conn.close();
            done_count[0] += 1;
        }
        return;
    }

    // ---- Master ----
    volatile tt_l1_ptr uint32_t* link_grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(link_grant_addr);
    volatile tt_l1_ptr uint32_t* consumed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_counter_addr);
    const uint64_t data_ready_mcast =
        get_noc_multicast_addr(mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, data_ready_sem_addr);
    MulticastEndpoint md_mcast;

    uint32_t last_arrived[num_slots] = {};
    uint32_t sent[num_slots] = {};  // chunks delivered to each remote target
    uint32_t last_consumed = 0;
    uint32_t go_local = 0;
    uint32_t last_done = 0;
    constexpr uint32_t kAllSlots = num_slots == 32 ? 0xFFFFFFFFu : ((1u << num_slots) - 1u);

    while (true) {
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
        for (uint32_t c = 0; c < num_dir_conns; ++c) {
            dir_conns[c].open();
        }

        uint32_t pending = kAllSlots;
        // Own target: false = waiting for data, true = handed to the workers, waiting for acks.
        bool own_handed = false;
        while (pending != 0 && !stop) {
            for (uint32_t k = 0; k < num_slots; ++k) {
                if ((pending & (1u << k)) == 0) {
                    continue;
                }
                if (d2d_relay::terminated(termination_addr, terminate_value)) {
                    stop = true;
                    break;
                }
                volatile tt_l1_ptr uint32_t* arrived =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrived_base + k * word_stride);
                const uint64_t credit_addr =
                    get_noc_addr(exit_noc_x, exit_noc_y, exit_cable_acked_base + k * word_stride);

                if (k == own_slot) {
                    if (!own_handed) {
                        if (!d2d_relay::advanced(arrived[0], last_arrived[k], z_lanes)) {
                            continue;
                        }
                        last_arrived[k] += z_lanes;
                        if constexpr (metadata_enabled) {
                            noc.async_write_multicast(
                                CoreLocalMem<uint32_t>(md_base + k * metadata_span),
                                md_mcast,
                                metadata_size,
                                num_workers,
                                {},
                                {.noc_x_start = mcast_x_start,
                                 .noc_y_start = mcast_y_start,
                                 .noc_x_end = mcast_x_end,
                                 .noc_y_end = mcast_y_end,
                                 .addr = worker_md_addr});
                            noc.async_write_barrier();
                        }
                        noc_semaphore_inc_multicast(data_ready_mcast, 1, num_workers);
                        own_handed = true;
                        continue;
                    }
                    if (!d2d_relay::advanced(consumed[0], last_consumed, num_workers)) {
                        continue;
                    }
                    last_consumed += num_workers;
                    d2d_relay::fabric_atomic_inc(z_conn, credit_addr, false, credit_header);
                    pending &= ~(1u << k);
                    continue;
                }

                volatile tt_l1_ptr uint32_t* recv_acked =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_acked_base + k * word_stride);
                if (recv_acked[0] != sent[k]) {
                    continue;  // target still reading the previous chunk
                }
                if (!d2d_relay::advanced(arrived[0], last_arrived[k], z_lanes)) {
                    continue;
                }
                last_arrived[k] += z_lanes;

                const Target t = target_at(slot_args, k);
                const bool two_lanes = t.lanes > 1;
                if (two_lanes) {
                    go_slot[0] = k;
                    go_local += 1;
                    go_count[0] = go_local;
                }
                copy_to_target(
                    noc,
                    dir_conns[t.master_conn],
                    t,
                    0,
                    two_lanes ? half_pages : total_pages,
                    cb_l1_addr,
                    data_header,
                    inc_header,
                    metadata_enabled != 0,
                    md_base + k * metadata_span);
                if (two_lanes) {
                    while (!d2d_relay::advanced(done_count[0], last_done, 1)) {
                        invalidate_l1_cache();
                    }
                    last_done += 1;
                }
                sent[k] += 1;
                // Staging k has been read: the exit relay may send the next chunk.
                d2d_relay::fabric_atomic_inc(z_conn, credit_addr, false, credit_header);
                pending &= ~(1u << k);
            }
        }

        noc_async_atomic_barrier();
        z_conn.close();
        for (uint32_t c = 0; c < num_dir_conns; ++c) {
            dir_conns[c].close();
        }
        link_grant[0] = 0;
        if (stop) {
            break;
        }
    }
}
