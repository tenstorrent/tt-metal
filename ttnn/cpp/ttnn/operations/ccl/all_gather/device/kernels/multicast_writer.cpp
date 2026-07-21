// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>
#include <array>
#include <type_traits>
#include <utility>

#include "multicast_common.hpp"

using address_t = uint32_t;

void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_stripe = get_compile_time_arg_val(2);
    constexpr uint32_t output_page_stripe_jump = get_compile_time_arg_val(3);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t packet_size = get_compile_time_arg_val(6);
    constexpr bool load_balance_across_alt_routes = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t num_connections = get_compile_time_arg_val(8);
    constexpr bool do_init_barrier = get_compile_time_arg_val(9) != 0;
    constexpr bool receiver_l1_mode = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t receiver_slot_count = get_compile_time_arg_val(11);
    constexpr bool receiver_fused_notify = get_compile_time_arg_val(12) != 0;
    constexpr bool receiver_credit_enabled = get_compile_time_arg_val(13) != 0;
    constexpr bool receiver_window_credit = get_compile_time_arg_val(14) != 0;
    constexpr bool receiver_proactive_credit = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t receiver_credit_group_batches = get_compile_time_arg_val(16);
    constexpr bool receiver_send_payload = get_compile_time_arg_val(17) != 0;
    constexpr bool receiver_attribution = get_compile_time_arg_val(18) != 0;
    constexpr bool bank_owned_links = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t bank_owned_num_banks = get_compile_time_arg_val(20);
    constexpr uint32_t bank_owned_num_links = get_compile_time_arg_val(21);
    constexpr uint32_t bank_owned_coalesce_mask = get_compile_time_arg_val(22);
    constexpr uint32_t receiver_cores_per_link = get_compile_time_arg_val(23);
    constexpr bool ring_fast_control_atomics = get_compile_time_arg_val(24) != 0;
    constexpr bool explicit_ring_path = get_compile_time_arg_val(25) != 0;
    constexpr bool bank_owned_coalesce_local = (bank_owned_coalesce_mask & 2) != 0;
    constexpr auto output_tensor_args = TensorAccessorArgs<26>();

    constexpr bool enable_fabric = (num_connections > 0);
    constexpr uint32_t output_page_size = output_chunks_per_page * output_chunk_size;
    constexpr uint32_t outputs_per_cb_page = cb_page_size / output_chunk_size;
    static_assert(!bank_owned_links || bank_owned_num_links > 0);
    static_assert(!bank_owned_links || bank_owned_num_banks % bank_owned_num_links == 0);
    static_assert(!bank_owned_links || bank_owned_num_banks == NUM_DRAM_BANKS);
    static_assert(receiver_cores_per_link > 0);
    static_assert(receiver_credit_group_batches > 0);
    static_assert(receiver_credit_group_batches <= receiver_slot_count);

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t output_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_chunk_in_stripe_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_byte_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_byte_offset_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_output_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t device_idx = get_arg_val<uint32_t>(arg_idx++);
    const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t line_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_e_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_w_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_spine_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t line_hops_alt = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_e_hops_alt = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_w_hops_alt = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_spine_hops_alt = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t line_physical_direction = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_e_physical_direction = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_w_physical_direction = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t rect_spine_physical_direction = get_arg_val<uint32_t>(arg_idx++);
    FabricExplicitPath fabric_path{};
    FabricExplicitPath fabric_path_alt{};
    const uint32_t fabric_path_config = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fabric_path_alt_config = get_arg_val<uint32_t>(arg_idx++);
    fabric_path.length = fabric_path_config & 0xff;
    fabric_path.escape_hop = (fabric_path_config >> 8) & 0xff;
    fabric_path_alt.length = fabric_path_alt_config & 0xff;
    fabric_path_alt.escape_hop = (fabric_path_alt_config >> 8) & 0xff;
    for (uint32_t i = 0; i < fabric_explicit_path_word_count; ++i) {
        fabric_path.words[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (uint32_t i = 0; i < fabric_explicit_path_word_count; ++i) {
        fabric_path_alt.words[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    const uint8_t receiver_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t receiver_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const address_t receiver_buffer_base = get_arg_val<address_t>(arg_idx++);
    const address_t produced_sem = get_arg_val<address_t>(arg_idx++);
    const address_t credit_sem = get_arg_val<address_t>(arg_idx++);
    const address_t consumed_sem = get_arg_val<address_t>(arg_idx++);
    const uint32_t credit_wait_value = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bank_owned_link_index = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_source_page_stride = get_arg_val<uint32_t>(arg_idx++);
    address_t credit_sems[receiver_cores_per_link];
    address_t consumed_sems[receiver_cores_per_link];
    uint8_t receiver_noc_xs[receiver_cores_per_link];
    uint8_t receiver_noc_ys[receiver_cores_per_link];
    credit_sems[0] = credit_sem;
    consumed_sems[0] = consumed_sem;
    receiver_noc_xs[0] = receiver_noc_x;
    receiver_noc_ys[0] = receiver_noc_y;
    for (uint32_t receiver_idx = 1; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
        credit_sems[receiver_idx] = get_arg_val<address_t>(arg_idx++);
        consumed_sems[receiver_idx] = get_arg_val<address_t>(arg_idx++);
        receiver_noc_xs[receiver_idx] = get_arg_val<uint32_t>(arg_idx++);
        receiver_noc_ys[receiver_idx] = get_arg_val<uint32_t>(arg_idx++);
    }
    size_t arg_for_fab = arg_idx;

    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    uint64_t init_barrier_cycles = 0;
    uint64_t cb_wait_cycles = 0;
    uint64_t local_issue_cycles = 0;
    uint64_t local_flush_cycles = 0;
    uint64_t credit_cycles = 0;
    uint64_t fabric_issue_cycles = 0;
    uint64_t fabric_flush_cycles = 0;
    uint64_t completion_cycles = 0;
    uint64_t local_write_command_count = 0;
    uint64_t fabric_payload_command_count = 0;
    uint64_t credit_command_count = 0;
    auto attribution_timestamp = []() __attribute__((always_inline)) -> uint32_t {
        if constexpr (receiver_attribution) {
            return get_timestamp_32b();
        }
        return 0;
    };
    auto attribute_elapsed = [&](uint64_t& accumulator, uint32_t start) __attribute__((always_inline)) {
        if constexpr (receiver_attribution) {
            accumulator += get_timestamp_32b() - start;
        }
    };

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    if constexpr (enable_fabric) {
        open_connections(fabric_connection, num_connections, arg_for_fab);
    }

    // Build ranges + ranges_alt arrays.
    // Connection order matches host: line (W) first, then rect (N) — only active ones are present,
    // indexed 0..num_connections-1.
    FabricRange ranges[2] = {};      // [0] = W-line; [1] = N-rect (Fabric_2D only)
    FabricRange ranges_alt[2] = {};  // [0] = W-line; [1] = N-rect (Fabric_2D only)
#ifdef FABRIC_2D
    {
        auto set_physical_direction = [](FabricRange& range, uint8_t direction, uint8_t hops) {
            if (direction == static_cast<uint8_t>(tt::tt_fabric::eth_chan_directions::EAST)) {
                range.e = hops;
            } else if (direction == static_cast<uint8_t>(tt::tt_fabric::eth_chan_directions::WEST)) {
                range.w = hops;
            } else if (direction == static_cast<uint8_t>(tt::tt_fabric::eth_chan_directions::NORTH)) {
                range.n = hops;
            } else {
                range.s = hops;
            }
        };
        uint32_t idx = 0;
        if (line_hops > 0) {
            set_physical_direction(ranges[idx], line_physical_direction, line_hops);
            set_physical_direction(ranges_alt[idx], line_physical_direction, line_hops_alt);
            ++idx;
        }
        if (rect_spine_hops > 0) {
            set_physical_direction(ranges[idx], rect_e_physical_direction, rect_e_hops);
            set_physical_direction(ranges[idx], rect_w_physical_direction, rect_w_hops);
            set_physical_direction(ranges[idx], rect_spine_physical_direction, rect_spine_hops);
            set_physical_direction(ranges_alt[idx], rect_e_physical_direction, rect_e_hops_alt);
            set_physical_direction(ranges_alt[idx], rect_w_physical_direction, rect_w_hops_alt);
            set_physical_direction(ranges_alt[idx], rect_spine_physical_direction, rect_spine_hops_alt);
            ++idx;
        }
    }
#else
    // 1D: exactly one of (line_hops, rect_spine_hops) is nonzero — that's the active axis.
    ranges[0] = (line_hops != 0) ? line_hops : rect_spine_hops;
    ranges_alt[0] = (line_hops != 0) ? line_hops_alt : rect_spine_hops_alt;
#endif

    // Allocate header and set state for semaphore sends
    uint8_t sem_route_id = 0;
    if constexpr (enable_fabric) {
        sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
        uint8_t starts[1] = {1};

        fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            fabric_connection,
            sem_route_id,
#ifndef FABRIC_2D
            starts,
#endif
            ranges,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                0u,    // ignore
                1u});  // increment 1
        apply_explicit_fabric_path<explicit_ring_path>(sem_route_id, fabric_path);
    }

    // Initialization barrier:
    // In some cases we don't have a guarantee that the output tensor has been allocated
    // on remote devices (every device's command queue executes asynchronously). So we wait
    // for this kernel to begin execution on all remote devices before sending any data.
    //
    // Mechanism:
    // Each worker core syncs with its mirror core (the same core) on all remote devices.
    // Reader fires sem increment forward, and also owns sem wait + decrement.
    // Writer fires sem increment backward, and implicitly gets blocked waiting for CB to
    // contain valid data.
    if constexpr ((explicit_ring_path && receiver_l1_mode) || do_init_barrier) {
        const uint32_t interval_start = attribution_timestamp();
        if constexpr (receiver_l1_mode) {
            // The local-source produced semaphore is never consumed by the
            // receiver batch loop, because the sender writes its own output
            // directly.  Reuse it as an explicit start handshake so the
            // receiver cannot publish readiness before this RISC has reset the
            // shared sender epoch.  Without the handshake, a fast receiver can
            // increment barrier_sem before the local sender is ready.  Do not
            // reset barrier_sem here: remote devices may already have sent
            // their initialization increments, and the reader retires the
            // complete epoch with an atomic decrement.
            for (uint32_t receiver_idx = 0; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_sems[receiver_idx]), 0);
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sems[receiver_idx]), 0);
            }
            if constexpr (receiver_cores_per_link == 1) {
                noc_semaphore_inc(safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem), 1);
            }
            for (uint32_t receiver_idx = 0; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
                noc_semaphore_inc(
                    safe_get_noc_addr(receiver_noc_xs[receiver_idx], receiver_noc_ys[receiver_idx], produced_sem), 1);
            }
            noc.async_atomic_barrier();
            if constexpr (receiver_cores_per_link > 1) {
                // Each receiver publishes readiness through its own consumed
                // counter.  Clear those temporary tokens before payload credit
                // epochs begin, then release the local reader through counter 0.
                for (uint32_t receiver_idx = 0; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
                    auto* ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sems[receiver_idx]);
                    noc_semaphore_wait_min(ready_sem, 1);
                    noc_semaphore_set(ready_sem, 0);
                }
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sems[0]), 1);
            } else {
                // The second local signal comes from the receiver after it resets all
                // produced counters.  Only then may this RISC advertise readiness remotely.
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 2);
            }
        }
        if constexpr (do_init_barrier) {
            if constexpr (enable_fabric) {
                uint64_t barrier_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
                fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_connection,
                    sem_route_id,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
            }
        }
        attribute_elapsed(init_barrier_cycles, interval_start);
    }

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    // "iterator" for output_tensor:
    //   byte_offset++ within an output page -> chunk++ -> stripe+=jump
    // (see the "Page indexing" glossary in all_gather_multicast_factory.cpp)
    // Returns {output_page_id, byte_offset} for the current chunk, then advances.
    // Supports any gather dim, any N-D shape, any shard spec.
    uint32_t output_page_id = output_page_id_start;
    uint32_t output_page_byte_off = output_page_byte_offset_start;
    uint32_t output_chunks_sent = 0;
    uint32_t output_chunk_in_stripe = output_chunk_in_stripe_start;
    const uint32_t bank_owned_pages_per_bank = bank_owned_links ? output_source_page_stride / bank_owned_num_banks : 0;
    const uint32_t bank_owned_runs_per_bank =
        bank_owned_links ? (bank_owned_pages_per_bank + outputs_per_cb_page - 1) / outputs_per_cb_page : 1;
    uint32_t bank_owned_output_batch = 0;
    uint32_t bank_owned_output_page_in_batch = 0;
    auto bank_owned_pages_in_batch = [&](uint32_t batch) __attribute__((always_inline)) {
        const uint32_t run_in_bank =
            receiver_cores_per_link > 1 ? batch / receiver_cores_per_link : batch % bank_owned_runs_per_bank;
        return std::min(outputs_per_cb_page, bank_owned_pages_per_bank - run_in_bank * outputs_per_cb_page);
    };
    auto valid_output_chunk = [&]() __attribute__((always_inline)) { return output_chunks_sent < num_output_chunks; };
    auto next_output_chunk = [&]() __attribute__((always_inline)) {
        if constexpr (bank_owned_links) {
            const uint32_t batch = bank_owned_output_batch;
            const uint32_t page_in_run = bank_owned_output_page_in_batch;
            const uint32_t owned_bank_slot =
                receiver_cores_per_link > 1 ? batch % receiver_cores_per_link : batch / bank_owned_runs_per_bank;
            const uint32_t run_in_bank =
                receiver_cores_per_link > 1 ? batch / receiver_cores_per_link : batch % bank_owned_runs_per_bank;
            const uint32_t bank = bank_owned_link_index + owned_bank_slot * bank_owned_num_links;
            const uint32_t page_id = device_idx * output_source_page_stride + bank +
                                     (run_in_bank * outputs_per_cb_page + page_in_run) * bank_owned_num_banks;
            ++output_chunks_sent;
            if (++bank_owned_output_page_in_batch == bank_owned_pages_in_batch(batch)) {
                bank_owned_output_page_in_batch = 0;
                ++bank_owned_output_batch;
            }
            return std::pair<uint32_t, uint32_t>{page_id, 0};
        }
        std::pair<uint32_t, uint32_t> loc{output_page_id, output_page_byte_off};
        output_chunks_sent++;
        if (++output_chunk_in_stripe == output_chunks_per_stripe) {
            output_chunk_in_stripe = 0;
            output_page_id += output_page_stripe_jump;
            output_page_byte_off = output_page_byte_offset;
        } else {
            output_page_byte_off += output_chunk_size;
            if (output_page_byte_off == output_page_size) {
                output_page_byte_off = 0;
                output_page_id++;
            }
        }
        return loc;
    };

    uint32_t receiver_batches_sent[receiver_cores_per_link] = {};
    uint32_t credit_groups_proxied[receiver_cores_per_link] = {};
    auto send_control_credit = [&](uint64_t noc_addr) __attribute__((always_inline)) {
        if constexpr (ring_fast_control_atomics) {
            fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<
                UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Flush>(
                fabric_connection, sem_route_id, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_addr, 0, false});
        } else {
            fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_connection, sem_route_id, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_addr, 0});
        }
    };
    auto proxy_ready_credit_groups = [&](uint32_t receiver_idx) __attribute__((always_inline)) {
        if constexpr (receiver_l1_mode && enable_fabric && receiver_credit_enabled && receiver_proactive_credit) {
            const uint32_t interval_start = attribution_timestamp();
            volatile tt_l1_ptr uint32_t* consumed_sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sems[receiver_idx]);
            const uint64_t remote_credit_sem_addr =
                safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, credit_sems[receiver_idx], 0);
            const uint32_t ready_groups = __atomic_load_n(consumed_sem_ptr, __ATOMIC_ACQUIRE);
            while (credit_groups_proxied[receiver_idx] < ready_groups) {
                send_control_credit(remote_credit_sem_addr);
                ++credit_groups_proxied[receiver_idx];
                if constexpr (receiver_attribution) {
                    ++credit_command_count;
                }
            }
            attribute_elapsed(credit_cycles, interval_start);
        }
    };
    auto prepare_receiver_slot = [&](uint32_t receiver_idx) __attribute__((always_inline)) -> uint32_t {
        if constexpr (receiver_l1_mode && enable_fabric && receiver_credit_enabled) {
            const uint32_t batches_sent = receiver_batches_sent[receiver_idx];
            volatile tt_l1_ptr uint32_t* consumed_sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sems[receiver_idx]);
            const uint64_t remote_credit_sem_addr =
                safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, credit_sems[receiver_idx], 0);
            if constexpr (receiver_proactive_credit) {
                proxy_ready_credit_groups(receiver_idx);
                if (batches_sent >= receiver_slot_count) {
                    const uint32_t reclaim_sequence =
                        (batches_sent - receiver_slot_count) / receiver_credit_group_batches + 1;
                    if (credit_groups_proxied[receiver_idx] < reclaim_sequence) {
                        const uint32_t interval_start = attribution_timestamp();
                        noc_semaphore_wait_min(consumed_sem_ptr, reclaim_sequence);
                        attribute_elapsed(credit_cycles, interval_start);
                        proxy_ready_credit_groups(receiver_idx);
                    }
                    const uint32_t interval_start = attribution_timestamp();
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_sems[receiver_idx]),
                        reclaim_sequence * credit_wait_value);
                    attribute_elapsed(credit_cycles, interval_start);
                }
                return batches_sent % receiver_slot_count;
            }
            uint32_t reclaim_sequence = 0;
            if constexpr (receiver_window_credit) {
                if (batches_sent > 0 && batches_sent % receiver_slot_count == 0) {
                    reclaim_sequence = batches_sent / receiver_slot_count;
                }
            } else if (batches_sent >= receiver_slot_count) {
                reclaim_sequence = batches_sent - receiver_slot_count + 1;
            }
            if (reclaim_sequence > 0) {
                const uint32_t interval_start = attribution_timestamp();
                noc_semaphore_wait_min(consumed_sem_ptr, reclaim_sequence);
                send_control_credit(remote_credit_sem_addr);
                if constexpr (receiver_attribution) {
                    ++credit_command_count;
                }
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_sems[receiver_idx]),
                    reclaim_sequence * credit_wait_value);
                attribute_elapsed(credit_cycles, interval_start);
            }
            return batches_sent % receiver_slot_count;
        }
        return 0u;
    };
    auto finish_receiver_batches = [&]() __attribute__((always_inline)) {
        if constexpr (receiver_l1_mode && enable_fabric && receiver_credit_enabled) {
            for (uint32_t receiver_idx = 0; receiver_idx < receiver_cores_per_link; ++receiver_idx) {
                volatile tt_l1_ptr uint32_t* consumed_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sems[receiver_idx]);
                if constexpr (receiver_proactive_credit) {
                    const uint32_t final_sequence =
                        (receiver_batches_sent[receiver_idx] + receiver_credit_group_batches - 1) /
                        receiver_credit_group_batches;
                    while (credit_groups_proxied[receiver_idx] < final_sequence) {
                        const uint32_t interval_start = attribution_timestamp();
                        noc_semaphore_wait_min(consumed_sem_ptr, credit_groups_proxied[receiver_idx] + 1);
                        attribute_elapsed(credit_cycles, interval_start);
                        proxy_ready_credit_groups(receiver_idx);
                    }
                    continue;
                }
                const uint32_t interval_start = attribution_timestamp();
                const uint32_t consumed_sequence =
                    receiver_window_credit
                        ? (receiver_batches_sent[receiver_idx] + receiver_slot_count - 1) / receiver_slot_count
                        : receiver_batches_sent[receiver_idx];
                noc_semaphore_wait_min(consumed_sem_ptr, consumed_sequence);
                attribute_elapsed(credit_cycles, interval_start);
            }
        }
    };
    auto run_main = [&](auto& fabric) {
        while (valid_output_chunk()) {
            const uint32_t cb_wait_start = attribution_timestamp();
            cb.wait_front(1);
            attribute_elapsed(cb_wait_cycles, cb_wait_start);
            auto l1_read_addr = cb.get_read_ptr();
            const auto batch_l1_read_addr = l1_read_addr;
            const uint32_t output_batch = bank_owned_output_batch;
            const uint32_t batch = bank_owned_links
                                       ? bank_owned_pages_in_batch(bank_owned_output_batch)
                                       : std::min(outputs_per_cb_page, num_output_chunks - output_chunks_sent);

            const uint32_t local_issue_start = attribution_timestamp();
            if constexpr (bank_owned_coalesce_local) {
                auto [page_id, page_byte_offset] = next_output_chunk();
                const uint64_t output_noc_addr = output_tensor_accessor.get_noc_addr(page_id, page_byte_offset);
                for (uint32_t i = 1; i < batch; ++i) {
                    next_output_chunk();
                }
                noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC>(
                    CoreLocalMem<uint32_t>(l1_read_addr),
                    UnicastEndpoint{},
                    batch * output_chunk_size,
                    {},
                    {.noc_x = static_cast<uint32_t>(NOC_UNICAST_ADDR_X(output_noc_addr)),
                     .noc_y = static_cast<uint32_t>(NOC_UNICAST_ADDR_Y(output_noc_addr)),
                     .addr = static_cast<uint32_t>(NOC_LOCAL_ADDR_OFFSET(output_noc_addr))},
                    {.vc = NOC_UNICAST_WRITE_VC + 1});
                if constexpr (receiver_attribution) {
                    ++local_write_command_count;
                }
                l1_read_addr += batch * output_chunk_size;
            } else {
                for (uint32_t i = 0; i < batch; ++i) {
                    auto [page_id, page_byte_offset] = next_output_chunk();
                    if constexpr (enable_fabric && !receiver_l1_mode) {
                        auto fabric_tensor_page_addr = tt::tt_fabric::addrgen_detail::get_noc_address(
                            output_tensor_accessor, page_id, page_byte_offset);
                        fabric.async_write(l1_read_addr, fabric_tensor_page_addr);
                    }

                    // For local writes use posted writes (to skip waiting for ack) on a different virtual channel
                    // so they do not interfere with Fabric writes on the same NOC.
                    noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC>(
                        CoreLocalMem<uint32_t>(l1_read_addr),
                        output_tensor_accessor,
                        output_chunk_size,
                        {},
                        {.page_id = page_id, .offset_bytes = page_byte_offset},
                        {.vc = NOC_UNICAST_WRITE_VC + 1});
                    if constexpr (receiver_attribution) {
                        ++local_write_command_count;
                    }

                    l1_read_addr += output_chunk_size;
                }
            }
            attribute_elapsed(local_issue_cycles, local_issue_start);

            if constexpr (enable_fabric && receiver_l1_mode && receiver_send_payload) {
                const uint32_t receiver_idx =
                    receiver_cores_per_link == 1 ? 0 : output_batch % receiver_cores_per_link;
                const uint32_t receiver_slot = prepare_receiver_slot(receiver_idx);
                const uint64_t remote_l1_addr = safe_get_noc_addr(
                    receiver_noc_xs[receiver_idx],
                    receiver_noc_ys[receiver_idx],
                    receiver_buffer_base + (device_idx * receiver_slot_count + receiver_slot) * cb_page_size,
                    0);
                const uint64_t remote_produced_sem_addr =
                    safe_get_noc_addr(receiver_noc_xs[receiver_idx], receiver_noc_ys[receiver_idx], produced_sem, 0);
                const uint32_t fabric_issue_start = attribution_timestamp();
                fabric.async_write(
                    batch_l1_read_addr, batch * output_chunk_size, remote_l1_addr, remote_produced_sem_addr);
                if constexpr (receiver_attribution) {
                    ++fabric_payload_command_count;
                }
                attribute_elapsed(fabric_issue_cycles, fabric_issue_start);
                ++receiver_batches_sent[receiver_idx];
            }

            const uint32_t local_flush_start = attribution_timestamp();
            noc.async_writes_flushed<NocOptions::POSTED>();  // wait for local writes to depart
            attribute_elapsed(local_flush_cycles, local_flush_start);
            if constexpr (enable_fabric) {
                const uint32_t fabric_flush_start = attribution_timestamp();
                fabric.async_writes_flushed();  // wait for Fabric writes
                if constexpr (!receiver_l1_mode || receiver_send_payload) {
                    attribute_elapsed(fabric_flush_cycles, fabric_flush_start);
                }
            }
            cb.pop_front(1);
        }
    };

    if constexpr (receiver_l1_mode) {
        FabricL1Writer<packet_size, load_balance_across_alt_routes, receiver_fused_notify, explicit_ring_path> fabric(
            noc, fabric_connection, num_connections, ranges, ranges_alt, fabric_path, fabric_path_alt);
        run_main(fabric);
    } else {
        FabricWriter<output_chunk_size, packet_size, load_balance_across_alt_routes, explicit_ring_path> fabric(
            noc, fabric_connection, num_connections, ranges, ranges_alt, fabric_path, fabric_path_alt);
        run_main(fabric);
    }

    finish_receiver_batches();

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Completion barrier:
    // We must only exit this op after guaranteeing that all remote data has arrived.
    //
    // Mechanism:
    // Each worker core sends a sem to its mirror core (the same core) on all remote devices. The sem
    // is sent after all data sends on a particular link, so it's correctly ordered at the receiver.
    // Reader fires sem increment forward, and also owns sem wait + decrement.
    // Writer fires sem increment backward, and exits immediately.
    const uint32_t completion_start = attribution_timestamp();
    if constexpr (enable_fabric) {
        uint64_t barrier_sem_noc_addr_in_pkt =
            safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection,
            sem_route_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
    }

    if constexpr (enable_fabric) {
        close_connections(fabric_connection);
    }
    noc.async_write_barrier();
    attribute_elapsed(completion_cycles, completion_start);

    if constexpr (receiver_attribution) {
        DeviceTimestampedData("AG_WRITER_INIT_BARRIER_CYCLES", init_barrier_cycles);
        DeviceTimestampedData("AG_WRITER_CB_WAIT_CYCLES", cb_wait_cycles);
        DeviceTimestampedData("AG_WRITER_LOCAL_ISSUE_CYCLES", local_issue_cycles);
        DeviceTimestampedData("AG_WRITER_LOCAL_FLUSH_CYCLES", local_flush_cycles);
        DeviceTimestampedData("AG_WRITER_CREDIT_CYCLES", credit_cycles);
        DeviceTimestampedData("AG_WRITER_FABRIC_ISSUE_CYCLES", fabric_issue_cycles);
        DeviceTimestampedData("AG_WRITER_FABRIC_FLUSH_CYCLES", fabric_flush_cycles);
        DeviceTimestampedData("AG_WRITER_COMPLETION_CYCLES", completion_cycles);
        DeviceTimestampedData("AG_WRITER_LOCAL_WRITE_COMMAND_COUNT", local_write_command_count);
        DeviceTimestampedData("AG_WRITER_FABRIC_PAYLOAD_COMMAND_COUNT", fabric_payload_command_count);
        DeviceTimestampedData("AG_WRITER_CREDIT_COMMAND_COUNT", credit_command_count);
    }
}
