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
#include <utility>

#include "multicast_common.hpp"

using address_t = uint32_t;

void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t input_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t output_chunks_per_stripe = get_compile_time_arg_val(3);
    constexpr uint32_t output_page_stripe_jump = get_compile_time_arg_val(4);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_depth = get_compile_time_arg_val(6);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t packet_size = get_compile_time_arg_val(8);
    constexpr bool load_balance_across_alt_routes = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t num_connections = get_compile_time_arg_val(10);
    constexpr bool do_init_barrier = get_compile_time_arg_val(11) != 0;
    constexpr bool receiver_l1_mode = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t receiver_slot_count = get_compile_time_arg_val(13);
    constexpr bool receiver_fused_notify = get_compile_time_arg_val(14) != 0;
    constexpr bool receiver_credit_enabled = get_compile_time_arg_val(15) != 0;
    constexpr bool receiver_window_credit = get_compile_time_arg_val(16) != 0;
    constexpr bool receiver_send_payload = get_compile_time_arg_val(17) != 0;
    constexpr bool receiver_attribution = get_compile_time_arg_val(18) != 0;
    constexpr bool receiver_address_attribution = get_compile_time_arg_val(19) != 0;
    constexpr bool bank_owned_links = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t bank_owned_num_banks = get_compile_time_arg_val(21);
    constexpr uint32_t bank_owned_num_links = get_compile_time_arg_val(22);
    constexpr uint32_t bank_owned_coalesce_mask = get_compile_time_arg_val(23);
    constexpr bool bank_owned_coalesce_source = (bank_owned_coalesce_mask & 1) != 0;
    constexpr auto input_tensor_args = TensorAccessorArgs<24>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr bool enable_fabric = (num_connections > 0);
    constexpr uint32_t output_page_size = output_chunks_per_page * output_chunk_size;
    constexpr uint32_t inputs_per_cb_page = cb_page_size / input_page_size;
    constexpr uint32_t outputs_per_cb_page = cb_page_size / output_chunk_size;
    static_assert(!bank_owned_links || bank_owned_num_links > 0);
    static_assert(!bank_owned_links || bank_owned_num_banks % bank_owned_num_links == 0);
    static_assert(!bank_owned_links || bank_owned_num_banks == NUM_DRAM_BANKS);

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t input_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_chunk_in_stripe_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_byte_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_byte_offset_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_output_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t device_idx = get_arg_val<uint32_t>(arg_idx++);
    const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_wait_value = get_arg_val<uint32_t>(arg_idx++);
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
    const uint8_t receiver_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t receiver_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const address_t receiver_buffer_base = get_arg_val<address_t>(arg_idx++);
    const address_t produced_sem = get_arg_val<address_t>(arg_idx++);
    const address_t credit_sem = get_arg_val<address_t>(arg_idx++);
    const address_t consumed_sem = get_arg_val<address_t>(arg_idx++);
    const uint32_t credit_wait_value = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bank_owned_link_index = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    uint64_t init_barrier_cycles = 0;
    uint64_t source_issue_cycles = 0;
    uint64_t source_wait_cycles = 0;
    uint64_t credit_cycles = 0;
    uint64_t fabric_cycles = 0;
    uint64_t completion_cycles = 0;
    uint64_t source_read_command_count = 0;
    uint64_t fabric_payload_command_count = 0;
    uint64_t credit_command_count = 0;
    uint64_t source_logical_adjacent_count = 0;
    uint64_t source_same_bank_adjacent_count = 0;
    uint64_t source_contiguous_adjacent_count = 0;
    uint64_t source_bank_predecessor_count = 0;
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
    // Connection order matches host: line (E) first, then rect (S) — only active ones are present,
    // indexed 0..num_connections-1.
    FabricRange ranges[2] = {};      // [0] = E-line, [1] = S-rect (Fabric_2D only)
    FabricRange ranges_alt[2] = {};  // [0] = E-line, [1] = S-rect (Fabric_2D only)
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
    if constexpr (do_init_barrier) {
        const uint32_t interval_start = attribution_timestamp();
        if constexpr (receiver_l1_mode) {
            // Local writer reset the credit/consumed epochs and the receiver
            // reset every produced epoch before these two readiness signals.
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 2);
        }
        if constexpr (enable_fabric) {
            uint64_t barrier_sem_noc_addr_in_pkt =
                safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
            fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_connection,
                sem_route_id,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), barrier_wait_value);
        // Atomic decrement (add -value), not reset to 0, so any increments from other phases are preserved.
        noc_semaphore_inc(
            safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem),
            (uint32_t)(-(int32_t)barrier_wait_value));
        attribute_elapsed(init_barrier_cycles, interval_start);
    }

    uint32_t receiver_batches_sent = 0;
    auto prepare_receiver_slot = [&]() __attribute__((always_inline)) -> uint32_t {
        if constexpr (receiver_l1_mode && enable_fabric && receiver_credit_enabled) {
            uint32_t reclaim_sequence = 0;
            if constexpr (receiver_window_credit) {
                if (receiver_batches_sent > 0 && receiver_batches_sent % receiver_slot_count == 0) {
                    reclaim_sequence = receiver_batches_sent / receiver_slot_count;
                }
            } else if (receiver_batches_sent >= receiver_slot_count) {
                reclaim_sequence = receiver_batches_sent - receiver_slot_count + 1;
            }
            if (reclaim_sequence > 0) {
                const uint32_t interval_start = attribution_timestamp();
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sem), reclaim_sequence);
                const uint64_t remote_credit_sem_addr =
                    safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, credit_sem, 0);
                fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_connection,
                    sem_route_id,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_credit_sem_addr, 0});
                if constexpr (receiver_attribution) {
                    ++credit_command_count;
                }
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_sem), reclaim_sequence * credit_wait_value);
                attribute_elapsed(credit_cycles, interval_start);
            }
            return receiver_batches_sent % receiver_slot_count;
        }
        return 0u;
    };
    auto finish_receiver_batches = [&]() __attribute__((always_inline)) {
        if constexpr (receiver_l1_mode && receiver_credit_enabled) {
            const uint32_t interval_start = attribution_timestamp();
            const uint32_t consumed_sequence =
                receiver_window_credit ? (receiver_batches_sent + receiver_slot_count - 1) / receiver_slot_count
                                       : receiver_batches_sent;
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sem), consumed_sequence);
            attribute_elapsed(credit_cycles, interval_start);
        }
    };

    auto run_main = [&](auto& fabric) {
        ///////////////////////////////////////////////////
        // MAIN
        ///////////////////////////////////////////////////

        // NOC transaction IDs to cycle between, and vars to keep track of state
        constexpr uint32_t max_trid = cb_depth;
        static_assert(max_trid <= NOC_MAX_TRANSACTION_ID, "max_trid exceeds max supported value");
        uint32_t curr_trid = 1;
        uint32_t wait_trid = 1;
        bool txns_in_flight = false;

        // Get write pointer (to write to CB) and read pointer (to read from CB).
        // We need to manually keep track of these pointers since we don't push_back
        // after every reserve_back when using NOC transaction IDs, so get_read/write_ptr()
        // will return stale values.
        auto l1_base_addr = cb.get_write_ptr();
        auto l1_end_addr = l1_base_addr + (cb_depth * cb_page_size);
        auto l1_write_addr = l1_base_addr;
        auto l1_read_addr = l1_base_addr;
        const UnicastEndpoint source_endpoint;

        // "iterator" for input_tensor.  The bank-owned variant enumerates the
        // complete source interval by link-owned banks, so each CB entry contains
        // one physical-bank run without an L1 permutation.
        uint32_t input_pages_read = 0;
        const uint32_t input_pages_in_range = input_page_id_end - input_page_id_start;
        const uint32_t worker_input_pages =
            bank_owned_links ? input_pages_in_range / bank_owned_num_links : input_pages_in_range;
        const uint32_t pages_per_bank = bank_owned_links ? input_pages_in_range / bank_owned_num_banks : 0;
        const uint32_t runs_per_bank =
            bank_owned_links ? (pages_per_bank + inputs_per_cb_page - 1) / inputs_per_cb_page : 1;
        uint32_t bank_owned_input_batch = 0;
        uint32_t bank_owned_input_page_in_batch = 0;
        constexpr uint32_t source_num_banks = NUM_DRAM_BANKS;
        bool have_previous_source_mapping = false;
        uint64_t previous_source_noc_addr = 0;
        auto valid_input_page_id = [&]()
                                       __attribute__((always_inline)) { return input_pages_read < worker_input_pages; };
        auto bank_owned_pages_in_batch = [&](uint32_t batch) __attribute__((always_inline)) {
            const uint32_t run_in_bank = batch % runs_per_bank;
            return std::min(inputs_per_cb_page, pages_per_bank - run_in_bank * inputs_per_cb_page);
        };
        auto next_input_page_id = [&]() __attribute__((always_inline)) {
            if constexpr (bank_owned_links) {
                const uint32_t batch = bank_owned_input_batch;
                const uint32_t page_in_run = bank_owned_input_page_in_batch;
                const uint32_t owned_bank_slot = batch / runs_per_bank;
                const uint32_t run_in_bank = batch % runs_per_bank;
                const uint32_t bank = bank_owned_link_index + owned_bank_slot * bank_owned_num_links;
                ++input_pages_read;
                if (++bank_owned_input_page_in_batch == bank_owned_pages_in_batch(batch)) {
                    bank_owned_input_page_in_batch = 0;
                    ++bank_owned_input_batch;
                }
                return input_page_id_start + bank +
                       (run_in_bank * inputs_per_cb_page + page_in_run) * bank_owned_num_banks;
            } else {
                return input_page_id_start + input_pages_read++;
            }
        };

        // "iterator" for output_tensor:
        //   byte_offset++ within an output page -> chunk++ -> stripe+=jump
        // (see the "Page indexing" glossary in all_gather_multicast_factory.cpp)
        // Returns {output_page_id, byte_offset} for the current chunk, then advances.
        // Supports any gather dim, any N-D shape, any shard spec.
        uint32_t output_page_id = output_page_id_start;
        uint32_t output_page_byte_off = output_page_byte_offset_start;
        uint32_t output_chunks_sent = 0;
        uint32_t output_batches_processed = 0;
        uint32_t output_chunk_in_stripe = output_chunk_in_stripe_start;
        auto valid_output_chunk = [&]()
                                      __attribute__((always_inline)) { return output_chunks_sent < num_output_chunks; };
        auto next_output_chunk = [&]() __attribute__((always_inline)) {
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

        // We reserve two to kick start the pipeline, and then it is steady state
        cb.reserve_back(2);
        while (valid_input_page_id()) {
            const uint32_t input_batch_pages =
                bank_owned_links ? bank_owned_pages_in_batch(bank_owned_input_batch)
                                 : std::min(inputs_per_cb_page, worker_input_pages - input_pages_read);
            // Read input tensor and fill CB page
            const uint32_t source_issue_start = attribution_timestamp();
            if constexpr (bank_owned_coalesce_source) {
                const uint32_t page_id = next_input_page_id();
                const uint64_t source_noc_addr = input_tensor_accessor.get_noc_addr(page_id);
                if constexpr (receiver_address_attribution) {
                    previous_source_noc_addr = source_noc_addr;
                    have_previous_source_mapping = true;
                    for (uint32_t i = 1; i < input_batch_pages; ++i) {
                        const uint32_t next_page_id = next_input_page_id();
                        const uint64_t next_source_noc_addr = input_tensor_accessor.get_noc_addr(next_page_id);
                        ++source_logical_adjacent_count;
                        if (NOC_UNICAST_ADDR_X(next_source_noc_addr) == NOC_UNICAST_ADDR_X(previous_source_noc_addr) &&
                            NOC_UNICAST_ADDR_Y(next_source_noc_addr) == NOC_UNICAST_ADDR_Y(previous_source_noc_addr)) {
                            ++source_same_bank_adjacent_count;
                            if (next_source_noc_addr == previous_source_noc_addr + input_page_size) {
                                ++source_contiguous_adjacent_count;
                            }
                        }
                        ++source_bank_predecessor_count;
                        previous_source_noc_addr = next_source_noc_addr;
                    }
                } else {
                    if (input_batch_pages > 1) {
                        input_pages_read += input_batch_pages - 1;
                        bank_owned_input_page_in_batch = 0;
                        ++bank_owned_input_batch;
                    }
                }
                noc.async_read<NocOptions::TXN_ID>(
                    source_endpoint,
                    CoreLocalMem<uint32_t>(l1_write_addr),
                    input_batch_pages * input_page_size,
                    {.noc_x = static_cast<uint32_t>(NOC_UNICAST_ADDR_X(source_noc_addr)),
                     .noc_y = static_cast<uint32_t>(NOC_UNICAST_ADDR_Y(source_noc_addr)),
                     .addr = static_cast<uint32_t>(NOC_LOCAL_ADDR_OFFSET(source_noc_addr))},
                    {},
                    {.trid = curr_trid});
                if constexpr (receiver_attribution) {
                    ++source_read_command_count;
                }
                l1_write_addr += cb_page_size;
            } else {
                for (uint32_t i = 0; i < input_batch_pages; ++i) {
                    auto page_id = next_input_page_id();
                    if constexpr (receiver_address_attribution) {
                        const uint64_t source_noc_addr = input_tensor_accessor.get_noc_addr(page_id);
                        if (have_previous_source_mapping) {
                            ++source_logical_adjacent_count;
                            if (NOC_UNICAST_ADDR_X(source_noc_addr) == NOC_UNICAST_ADDR_X(previous_source_noc_addr) &&
                                NOC_UNICAST_ADDR_Y(source_noc_addr) == NOC_UNICAST_ADDR_Y(previous_source_noc_addr)) {
                                ++source_same_bank_adjacent_count;
                                if (source_noc_addr == previous_source_noc_addr + input_page_size) {
                                    ++source_contiguous_adjacent_count;
                                }
                            }
                        }
                        if (page_id >= input_page_id_start + source_num_banks) {
                            const uint64_t bank_predecessor_noc_addr =
                                input_tensor_accessor.get_noc_addr(page_id - source_num_banks);
                            if (source_noc_addr == bank_predecessor_noc_addr + input_page_size) {
                                ++source_bank_predecessor_count;
                            }
                        }
                        previous_source_noc_addr = source_noc_addr;
                        have_previous_source_mapping = true;
                    }
                    noc.async_read<NocOptions::TXN_ID>(
                        input_tensor_accessor,
                        CoreLocalMem<uint32_t>(l1_write_addr),
                        input_page_size,
                        {.page_id = page_id},
                        {},
                        {.trid = curr_trid});
                    if constexpr (receiver_attribution) {
                        ++source_read_command_count;
                    }
                    l1_write_addr += input_page_size;
                }
                if constexpr (bank_owned_links) {
                    l1_write_addr += (inputs_per_cb_page - input_batch_pages) * input_page_size;
                }
            }
            attribute_elapsed(source_issue_cycles, source_issue_start);
            if (l1_write_addr == l1_end_addr) {
                l1_write_addr = l1_base_addr;
            }

            curr_trid = (curr_trid == max_trid) ? 1 : curr_trid + 1;
            if (txns_in_flight) {
                // push_back() will unblock the writer to send Fabric data in opposite dir
                const uint32_t source_wait_start = attribution_timestamp();
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = wait_trid});
                attribute_elapsed(source_wait_cycles, source_wait_start);
                cb.push_back(1);
                wait_trid = (wait_trid == max_trid) ? 1 : (wait_trid + 1);

                if constexpr (enable_fabric) {
                    uint32_t fabric_start = 0;
                    if constexpr (receiver_l1_mode) {
                        const uint32_t batch =
                            bank_owned_links ? bank_owned_pages_in_batch(output_batches_processed)
                                             : std::min(outputs_per_cb_page, num_output_chunks - output_chunks_sent);
                        if constexpr (receiver_send_payload) {
                            const uint32_t receiver_slot = prepare_receiver_slot();
                            fabric_start = attribution_timestamp();
                            const uint64_t remote_l1_addr = safe_get_noc_addr(
                                receiver_noc_x,
                                receiver_noc_y,
                                receiver_buffer_base +
                                    (device_idx * receiver_slot_count + receiver_slot) * cb_page_size,
                                0);
                            const uint64_t remote_produced_sem_addr =
                                safe_get_noc_addr(receiver_noc_x, receiver_noc_y, produced_sem, 0);
                            fabric.async_write(
                                l1_read_addr, batch * output_chunk_size, remote_l1_addr, remote_produced_sem_addr);
                            if constexpr (receiver_attribution) {
                                ++fabric_payload_command_count;
                            }
                            ++receiver_batches_sent;
                        }
                        output_chunks_sent += batch;
                        ++output_batches_processed;
                        l1_read_addr += batch * output_chunk_size;
                        if constexpr (bank_owned_links) {
                            l1_read_addr += (outputs_per_cb_page - batch) * output_chunk_size;
                        }
                    } else {
                        fabric_start = attribution_timestamp();
                        // Send Fabric data in our dir
                        for (uint32_t i = 0; i < outputs_per_cb_page && valid_output_chunk(); ++i) {
                            auto [page_id, page_byte_offset] = next_output_chunk();
                            auto fabric_tensor_page_addr = tt::tt_fabric::addrgen_detail::get_noc_address(
                                output_tensor_accessor, page_id, page_byte_offset);
                            fabric.async_write(l1_read_addr, fabric_tensor_page_addr);
                            l1_read_addr += output_chunk_size;
                        }
                    }
                    fabric.async_writes_flushed();
                    if constexpr (!receiver_l1_mode || receiver_send_payload) {
                        attribute_elapsed(fabric_cycles, fabric_start);
                    }
                    if (l1_read_addr == l1_end_addr) {
                        l1_read_addr = l1_base_addr;
                    }
                }

                // Reserve for next block.
                // Reserve back is not incremental, so to reserve one more, we need to reserve 2.
                // This accounts for the one we already have reserved (for in-flight read).
                cb.reserve_back(2);
            }
            txns_in_flight = true;
        }
        // Drain in-flight reads
        while (wait_trid != curr_trid) {
            // push_back() will unblock the writer to send Fabric data in opposite dir
            const uint32_t source_wait_start = attribution_timestamp();
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = wait_trid});
            attribute_elapsed(source_wait_cycles, source_wait_start);
            cb.push_back(1);
            wait_trid = (wait_trid == max_trid) ? 1 : (wait_trid + 1);

            if constexpr (enable_fabric) {
                uint32_t fabric_start = 0;
                if constexpr (receiver_l1_mode) {
                    const uint32_t batch = bank_owned_links
                                               ? bank_owned_pages_in_batch(output_batches_processed)
                                               : std::min(outputs_per_cb_page, num_output_chunks - output_chunks_sent);
                    if constexpr (receiver_send_payload) {
                        const uint32_t receiver_slot = prepare_receiver_slot();
                        fabric_start = attribution_timestamp();
                        const uint64_t remote_l1_addr = safe_get_noc_addr(
                            receiver_noc_x,
                            receiver_noc_y,
                            receiver_buffer_base + (device_idx * receiver_slot_count + receiver_slot) * cb_page_size,
                            0);
                        const uint64_t remote_produced_sem_addr =
                            safe_get_noc_addr(receiver_noc_x, receiver_noc_y, produced_sem, 0);
                        fabric.async_write(
                            l1_read_addr, batch * output_chunk_size, remote_l1_addr, remote_produced_sem_addr);
                        if constexpr (receiver_attribution) {
                            ++fabric_payload_command_count;
                        }
                        ++receiver_batches_sent;
                    }
                    output_chunks_sent += batch;
                    ++output_batches_processed;
                    l1_read_addr += batch * output_chunk_size;
                    if constexpr (bank_owned_links) {
                        l1_read_addr += (outputs_per_cb_page - batch) * output_chunk_size;
                    }
                } else {
                    fabric_start = attribution_timestamp();
                    // Send Fabric data in our dir
                    for (uint32_t i = 0; i < outputs_per_cb_page && valid_output_chunk(); ++i) {
                        auto [page_id, page_byte_offset] = next_output_chunk();
                        auto fabric_tensor_page_addr = tt::tt_fabric::addrgen_detail::get_noc_address(
                            output_tensor_accessor, page_id, page_byte_offset);
                        fabric.async_write(l1_read_addr, fabric_tensor_page_addr);
                        l1_read_addr += output_chunk_size;
                    }
                }
                fabric.async_writes_flushed();
                if constexpr (!receiver_l1_mode || receiver_send_payload) {
                    attribute_elapsed(fabric_cycles, fabric_start);
                }
                if (l1_read_addr == l1_end_addr) {
                    l1_read_addr = l1_base_addr;
                }
            }
        }
    };

    if constexpr (receiver_l1_mode) {
        FabricL1Writer<packet_size, load_balance_across_alt_routes, receiver_fused_notify> fabric(
            noc, fabric_connection, num_connections, ranges, ranges_alt);
        run_main(fabric);
    } else {
        FabricWriter<output_chunk_size, packet_size, load_balance_across_alt_routes> fabric(
            noc, fabric_connection, num_connections, ranges, ranges_alt);
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
    const uint32_t completion_wait_value = barrier_wait_value - (receiver_l1_mode ? 2 : 0);
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), completion_wait_value);
    // Atomic decrement (add -value), not reset to 0, so any increments from other phases are preserved.
    noc_semaphore_inc(
        safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem),
        (uint32_t)(-(int32_t)completion_wait_value));

    if constexpr (enable_fabric) {
        close_connections(fabric_connection);
    }
    noc.async_write_barrier();
    attribute_elapsed(completion_cycles, completion_start);

    if constexpr (receiver_attribution) {
        DeviceTimestampedData("AG_READER_INIT_BARRIER_CYCLES", init_barrier_cycles);
        DeviceTimestampedData("AG_READER_SOURCE_ISSUE_CYCLES", source_issue_cycles);
        DeviceTimestampedData("AG_READER_CREDIT_CYCLES", credit_cycles);
        DeviceTimestampedData("AG_READER_SOURCE_WAIT_CYCLES", source_wait_cycles);
        DeviceTimestampedData("AG_R_FABRIC_CYCLES", fabric_cycles);
        DeviceTimestampedData("AG_READER_COMPLETION_CYCLES", completion_cycles);
        DeviceTimestampedData("AG_READER_SOURCE_READ_COMMAND_COUNT", source_read_command_count);
        DeviceTimestampedData("AG_READER_FABRIC_PAYLOAD_COMMAND_COUNT", fabric_payload_command_count);
        DeviceTimestampedData("AG_READER_CREDIT_COMMAND_COUNT", credit_command_count);
    }
    if constexpr (receiver_address_attribution) {
        DeviceTimestampedData("AG_READER_SOURCE_LOGICAL_ADJACENT_COUNT", source_logical_adjacent_count);
        DeviceTimestampedData("AG_READER_SOURCE_SAME_BANK_ADJACENT_COUNT", source_same_bank_adjacent_count);
        DeviceTimestampedData("AG_READER_SOURCE_CONTIGUOUS_ADJACENT_COUNT", source_contiguous_adjacent_count);
        DeviceTimestampedData("AG_READER_SOURCE_BANK_PREDECESSOR_COUNT", source_bank_predecessor_count);
    }
}
