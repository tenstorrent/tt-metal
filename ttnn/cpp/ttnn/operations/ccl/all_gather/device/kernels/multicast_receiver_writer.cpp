// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

#include <cstdint>

using address_t = uint32_t;

// Windowed receiver for row-major height gather.  Every source owns disjoint
// source/slot L1 storage.  One or both data-movement RISCs drain disjoint batches;
// the BRISC publishes consumption only after both RISCs finish the whole window.
// The mirrored sender core proxies credits over existing Fabric routes, so neither
// receiver RISC opens a Fabric client.
void kernel_main() {
    constexpr uint32_t output_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t inputs_per_cb_page = get_compile_time_arg_val(2);
    constexpr uint32_t num_devices = get_compile_time_arg_val(3);
    constexpr uint32_t receiver_buffer_base = get_compile_time_arg_val(4);
    constexpr bool drain_to_output = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t receiver_slot_count = get_compile_time_arg_val(6);
    constexpr bool run_receiver_batches = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t receiver_credit_group_batches = get_compile_time_arg_val(8);
    constexpr bool wait_for_payload = get_compile_time_arg_val(9) != 0;
    constexpr bool receiver_attribution = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t receiver_drain_risc_count = get_compile_time_arg_val(11);
    constexpr uint32_t receiver_drain_risc_index = get_compile_time_arg_val(12);
    constexpr bool receiver_address_attribution = get_compile_time_arg_val(13) != 0;
    constexpr bool bank_owned_links = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t bank_owned_num_banks = get_compile_time_arg_val(15);
    constexpr uint32_t bank_owned_num_links = get_compile_time_arg_val(16);
    constexpr uint32_t bank_owned_coalesce_mask = get_compile_time_arg_val(17);
    constexpr uint32_t receiver_cores_per_link = get_compile_time_arg_val(18);
    constexpr bool active_axis_is_ring = get_compile_time_arg_val(19) != 0;
    constexpr bool bank_owned_coalesce_receiver = (bank_owned_coalesce_mask & 4) != 0;
    constexpr auto output_tensor_args = TensorAccessorArgs<20>();
    static_assert(receiver_drain_risc_count == 1 || receiver_drain_risc_count == 2);
    static_assert(receiver_drain_risc_index < receiver_drain_risc_count);
    static_assert(receiver_credit_group_batches > 0);
    static_assert(receiver_credit_group_batches <= receiver_slot_count);
    static_assert(!bank_owned_links || bank_owned_num_links > 0);
    static_assert(!bank_owned_links || bank_owned_num_banks % bank_owned_num_links == 0);
    static_assert(!bank_owned_links || bank_owned_num_banks == NUM_DRAM_BANKS);
    static_assert(receiver_cores_per_link > 0);

    size_t arg_idx = 0;
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t local_device_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t selected_input_page_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t selected_input_page_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_source_page_stride = get_arg_val<uint32_t>(arg_idx++);
    const address_t consumed_sem = get_arg_val<address_t>(arg_idx++);
    const uint8_t sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t sender_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const address_t barrier_sem = get_arg_val<address_t>(arg_idx++);
    const uint32_t bank_owned_link_index = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receiver_bank_slot = get_arg_val<uint32_t>(arg_idx++);

    address_t produced_sem_forward[num_devices];
    address_t produced_sem_backward[num_devices];
    for (uint32_t source = 0; source < num_devices; ++source) {
        produced_sem_forward[source] = get_arg_val<address_t>(arg_idx++);
    }
    for (uint32_t source = 0; source < num_devices; ++source) {
        produced_sem_backward[source] = get_arg_val<address_t>(arg_idx++);
    }
    const address_t dual_risc_sync_sem = get_arg_val<address_t>(arg_idx++);

    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);
    Noc noc;

    uint64_t init_cycles = 0;
    uint64_t produced_wait_cycles = 0;
    uint64_t drain_issue_cycles = 0;
    uint64_t drain_flush_cycles = 0;
    uint64_t consumed_publish_cycles = 0;
    uint64_t dual_risc_sync_cycles = 0;
    uint64_t completion_cycles = 0;
    uint64_t produced_wait_count = 0;
    uint64_t drain_write_command_count = 0;
    uint64_t consumed_publish_count = 0;
    uint64_t output_logical_adjacent_count = 0;
    uint64_t output_same_bank_adjacent_count = 0;
    uint64_t output_contiguous_adjacent_count = 0;
    uint64_t output_bank_predecessor_count = 0;
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

    // Establish a clean sequence-number epoch before advertising local receiver
    // readiness to the sender core.  The sender's init barrier prevents any payload
    // from being emitted until the mirrored receiver on every device is ready.
    volatile tt_l1_ptr uint32_t* dual_risc_sync_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dual_risc_sync_sem);
    const uint32_t init_start = attribution_timestamp();
    if constexpr (receiver_drain_risc_index == 0) {
        // The sender uses its otherwise-unused local-source produced semaphore
        // as a start handshake.  Waiting here orders this receiver's ready
        // signal after the sender resets the shared barrier/credit epoch.
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(produced_sem_backward[local_device_idx]), 1);
        for (uint32_t source = 0; source < num_devices; ++source) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(produced_sem_forward[source]), 0);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(produced_sem_backward[source]), 0);
        }
        if constexpr (receiver_drain_risc_count == 2) {
            // The previous invocation ends above 1.  Equality prevents the
            // second RISC from accepting stale produced counters before reset.
            __atomic_store_n(dual_risc_sync_sem_ptr, 1, __ATOMIC_RELEASE);
        }
        if constexpr (receiver_cores_per_link > 1) {
            noc_semaphore_inc(safe_get_noc_addr(sender_noc_x, sender_noc_y, consumed_sem), 1);
        } else {
            noc_semaphore_inc(safe_get_noc_addr(sender_noc_x, sender_noc_y, barrier_sem), 1);
        }
        noc.async_atomic_barrier();
    } else {
        while (__atomic_load_n(dual_risc_sync_sem_ptr, __ATOMIC_ACQUIRE) != 1) {
        }
    }
    attribute_elapsed(init_cycles, init_start);

    if constexpr (!run_receiver_batches) {
        if constexpr (receiver_attribution) {
            DeviceTimestampedData("AG_RECEIVER_INIT_CYCLES", init_cycles);
        }
        return;
    }

    const uint32_t worker_pages = selected_input_page_end - selected_input_page_start;
    constexpr uint32_t output_num_banks = NUM_DRAM_BANKS;
    const uint32_t bank_owned_pages_per_bank = bank_owned_links ? output_source_page_stride / bank_owned_num_banks : 0;
    const uint32_t bank_owned_runs_per_bank =
        bank_owned_links ? (bank_owned_pages_per_bank + inputs_per_cb_page - 1) / inputs_per_cb_page : 1;
    const uint32_t num_batches =
        bank_owned_links ? bank_owned_runs_per_bank *
                               (receiver_cores_per_link == 1 ? bank_owned_num_banks / bank_owned_num_links : 1)
                         : (worker_pages + inputs_per_cb_page - 1) / inputs_per_cb_page;
    auto bank_owned_pages_in_batch = [&](uint32_t batch) __attribute__((always_inline)) {
        // Each interleaved receiver core drains one bank, so its batch index is
        // already the run index within that bank.  Only the single-receiver
        // schedule interleaves several banks in this loop.
        const uint32_t run_in_bank = batch % bank_owned_runs_per_bank;
        return std::min(inputs_per_cb_page, bank_owned_pages_per_bank - run_in_bank * inputs_per_cb_page);
    };
    constexpr uint32_t batches_per_window = receiver_credit_group_batches;
    const uint32_t num_windows = (num_batches + batches_per_window - 1) / batches_per_window;

    for (uint32_t window = 0; window < num_windows; ++window) {
        const uint32_t batch_start = window * batches_per_window;
        const uint32_t batch_end = std::min(batch_start + batches_per_window, num_batches);
        for (uint32_t batch = batch_start + receiver_drain_risc_index; batch < batch_end;
             batch += receiver_drain_risc_count) {
            const uint32_t logical_bank_owned_batch =
                receiver_cores_per_link == 1 ? batch : batch * receiver_cores_per_link + receiver_bank_slot;
            const uint32_t page_in_worker = batch * inputs_per_cb_page;
            const uint32_t pages_in_batch = bank_owned_links
                                                ? bank_owned_pages_in_batch(batch)
                                                : std::min(inputs_per_cb_page, worker_pages - page_in_worker);

            for (uint32_t source = 0; source < num_devices; ++source) {
                if (source == local_device_idx) {
                    continue;
                }

                if constexpr (wait_for_payload) {
                    const uint32_t wait_start = attribution_timestamp();
                    address_t source_produced_sem = produced_sem_forward[source];
                    uint32_t produced_sequence = batch + 1;
                    if constexpr (bank_owned_links) {
                        if constexpr (active_axis_is_ring) {
                            const uint32_t forward_distance = (local_device_idx + num_devices - source) % num_devices;
                            const uint32_t half_ring = num_devices / 2;
                            if (forward_distance > half_ring ||
                                (forward_distance == half_ring && (logical_bank_owned_batch & 1) != 0)) {
                                source_produced_sem = produced_sem_backward[source];
                            }
                            if (num_devices % 2 == 0 && forward_distance == half_ring && receiver_cores_per_link == 1) {
                                produced_sequence = batch / 2 + 1;
                            }
                        } else if (source > local_device_idx) {
                            source_produced_sem = produced_sem_backward[source];
                        }
                    }
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_produced_sem), produced_sequence);
                    attribute_elapsed(produced_wait_cycles, wait_start);
                    if constexpr (receiver_attribution) {
                        ++produced_wait_count;
                    }
                }
                if constexpr (drain_to_output) {
                    uint32_t source_l1_addr =
                        receiver_buffer_base +
                        (source * receiver_slot_count + batch % receiver_slot_count) * cb_page_size;
                    uint32_t output_page_id =
                        source * output_source_page_stride + selected_input_page_start + page_in_worker;
                    if constexpr (bank_owned_links) {
                        const uint32_t owned_bank_slot =
                            receiver_cores_per_link == 1 ? batch / bank_owned_runs_per_bank : receiver_bank_slot;
                        const uint32_t run_in_bank = batch % bank_owned_runs_per_bank;
                        const uint32_t bank = bank_owned_link_index + owned_bank_slot * bank_owned_num_links;
                        output_page_id = source * output_source_page_stride + bank +
                                         run_in_bank * inputs_per_cb_page * bank_owned_num_banks;
                    }
                    const uint32_t drain_issue_start = attribution_timestamp();
                    if constexpr (bank_owned_coalesce_receiver) {
                        const uint64_t output_noc_addr = output_tensor_accessor.get_noc_addr(output_page_id);
                        if constexpr (receiver_address_attribution) {
                            uint64_t previous_output_noc_addr = output_noc_addr;
                            for (uint32_t page = 1; page < pages_in_batch; ++page) {
                                const uint32_t next_output_page_id = output_page_id + page * bank_owned_num_banks;
                                const uint64_t next_output_noc_addr =
                                    output_tensor_accessor.get_noc_addr(next_output_page_id);
                                ++output_logical_adjacent_count;
                                if (NOC_UNICAST_ADDR_X(next_output_noc_addr) ==
                                        NOC_UNICAST_ADDR_X(previous_output_noc_addr) &&
                                    NOC_UNICAST_ADDR_Y(next_output_noc_addr) ==
                                        NOC_UNICAST_ADDR_Y(previous_output_noc_addr)) {
                                    ++output_same_bank_adjacent_count;
                                    if (next_output_noc_addr == previous_output_noc_addr + output_page_size) {
                                        ++output_contiguous_adjacent_count;
                                    }
                                }
                                const uint64_t bank_predecessor_noc_addr =
                                    output_tensor_accessor.get_noc_addr(next_output_page_id - output_num_banks);
                                if (next_output_noc_addr == bank_predecessor_noc_addr + output_page_size) {
                                    ++output_bank_predecessor_count;
                                }
                                previous_output_noc_addr = next_output_noc_addr;
                            }
                        }
                        noc.async_write<NocOptions::POSTED>(
                            CoreLocalMem<uint32_t>(source_l1_addr),
                            UnicastEndpoint{},
                            pages_in_batch * output_page_size,
                            {},
                            {.noc_x = static_cast<uint32_t>(NOC_UNICAST_ADDR_X(output_noc_addr)),
                             .noc_y = static_cast<uint32_t>(NOC_UNICAST_ADDR_Y(output_noc_addr)),
                             .addr = static_cast<uint32_t>(NOC_LOCAL_ADDR_OFFSET(output_noc_addr))});
                        if constexpr (receiver_attribution) {
                            ++drain_write_command_count;
                        }
                    } else {
                        const uint32_t first_output_page_id = output_page_id;
                        bool have_previous_output_mapping = false;
                        uint64_t previous_output_noc_addr = 0;
                        for (uint32_t page = 0; page < pages_in_batch; ++page) {
                            if constexpr (receiver_address_attribution) {
                                const uint64_t output_noc_addr = output_tensor_accessor.get_noc_addr(output_page_id);
                                if (have_previous_output_mapping) {
                                    ++output_logical_adjacent_count;
                                    if (NOC_UNICAST_ADDR_X(output_noc_addr) ==
                                            NOC_UNICAST_ADDR_X(previous_output_noc_addr) &&
                                        NOC_UNICAST_ADDR_Y(output_noc_addr) ==
                                            NOC_UNICAST_ADDR_Y(previous_output_noc_addr)) {
                                        ++output_same_bank_adjacent_count;
                                        if (output_noc_addr == previous_output_noc_addr + output_page_size) {
                                            ++output_contiguous_adjacent_count;
                                        }
                                    }
                                }
                                if (output_page_id >= first_output_page_id + output_num_banks) {
                                    const uint64_t bank_predecessor_noc_addr =
                                        output_tensor_accessor.get_noc_addr(output_page_id - output_num_banks);
                                    if (output_noc_addr == bank_predecessor_noc_addr + output_page_size) {
                                        ++output_bank_predecessor_count;
                                    }
                                }
                                previous_output_noc_addr = output_noc_addr;
                                have_previous_output_mapping = true;
                            }
                            noc.async_write<NocOptions::POSTED>(
                                CoreLocalMem<uint32_t>(source_l1_addr),
                                output_tensor_accessor,
                                output_page_size,
                                {},
                                {.page_id = output_page_id});
                            if constexpr (receiver_attribution) {
                                ++drain_write_command_count;
                            }
                            source_l1_addr += output_page_size;
                            output_page_id += bank_owned_links ? bank_owned_num_banks : 1;
                        }
                    }
                    attribute_elapsed(drain_issue_cycles, drain_issue_start);
                }
            }

            // A flushed posted write has consumed this batch's L1 source bytes.
            if constexpr (drain_to_output) {
                const uint32_t drain_flush_start = attribution_timestamp();
                noc.async_writes_flushed<NocOptions::POSTED>();
                attribute_elapsed(drain_flush_cycles, drain_flush_start);
            }
        }

        if constexpr (receiver_drain_risc_count == 2) {
            const uint32_t sync_start = attribution_timestamp();
            if constexpr (receiver_drain_risc_index == 1) {
                __atomic_fetch_add(dual_risc_sync_sem_ptr, 1, __ATOMIC_RELEASE);
            } else {
                while (__atomic_load_n(dual_risc_sync_sem_ptr, __ATOMIC_ACQUIRE) < window + 2) {
                }
            }
            attribute_elapsed(dual_risc_sync_cycles, sync_start);
        }
        if constexpr (wait_for_payload && receiver_drain_risc_index == 0) {
            const uint32_t publish_start = attribution_timestamp();
            noc_semaphore_inc(safe_get_noc_addr(sender_noc_x, sender_noc_y, consumed_sem), 1);
            attribute_elapsed(consumed_publish_cycles, publish_start);
            if constexpr (receiver_attribution) {
                ++consumed_publish_count;
            }
        }
    }

    const uint32_t completion_start = attribution_timestamp();
    noc.async_atomic_barrier();
    if constexpr (drain_to_output) {
        noc.async_write_barrier();
    }
    attribute_elapsed(completion_cycles, completion_start);

    // Emit one aggregate per interval after all receiver work is complete.
    if constexpr (receiver_attribution) {
        DeviceTimestampedData("AG_RECEIVER_INIT_CYCLES", init_cycles);
        DeviceTimestampedData("AG_RECEIVER_PRODUCED_WAIT_CYCLES", produced_wait_cycles);
        DeviceTimestampedData("AG_RECEIVER_DRAIN_ISSUE_CYCLES", drain_issue_cycles);
        DeviceTimestampedData("AG_RECEIVER_DRAIN_FLUSH_CYCLES", drain_flush_cycles);
        DeviceTimestampedData("AG_RECEIVER_CONSUMED_PUBLISH_CYCLES", consumed_publish_cycles);
        DeviceTimestampedData("AG_RECEIVER_DUAL_SYNC_CYCLES", dual_risc_sync_cycles);
        DeviceTimestampedData("AG_RECEIVER_COMPLETION_CYCLES", completion_cycles);
        DeviceTimestampedData("AG_RECEIVER_PRODUCED_WAIT_COUNT", produced_wait_count);
        DeviceTimestampedData("AG_RECEIVER_DRAIN_WRITE_COMMAND_COUNT", drain_write_command_count);
        DeviceTimestampedData("AG_RECEIVER_CONSUMED_PUBLISH_COUNT", consumed_publish_count);
    }
    if constexpr (receiver_address_attribution) {
        DeviceTimestampedData("AG_RECEIVER_OUTPUT_LOGICAL_ADJACENT_COUNT", output_logical_adjacent_count);
        DeviceTimestampedData("AG_RECEIVER_OUTPUT_SAME_BANK_ADJACENT_COUNT", output_same_bank_adjacent_count);
        DeviceTimestampedData("AG_RECEIVER_OUTPUT_CONTIGUOUS_ADJACENT_COUNT", output_contiguous_adjacent_count);
        DeviceTimestampedData("AG_RECEIVER_OUTPUT_BANK_PREDECESSOR_COUNT", output_bank_predecessor_count);
    }
}
