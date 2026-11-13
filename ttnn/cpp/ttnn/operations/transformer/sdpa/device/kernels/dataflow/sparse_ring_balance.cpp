// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"  // PROFILE (local only)
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_ring_balance_plan.hpp"

// Load balancing of sparse_ring_sdpa, one core per device (next to the gather workers), every call:
//   1. wait until every SDPA core has written its static units' costs into this device's cost page (DRAM);
//   2. cross-device (remote): have the gather writers multicast the page to every device (flag 1), wait for the other
//      devices' pages, run the row-block swaps (every device computes all of them, the same way), write this device's
//      partner page and have the writers send its swapped row blocks (flag 2);
//   3. deal every head's units over its cores on the costs of the units the slots now hold
//      (sparse_balance::plan_unit_tables), write each SDPA core's unit table into its cb_balance page, then release it;
//   4. cross-device: once the swapped-in row blocks have arrived, tell every SDPA core; once their outputs are
//      written, have the writers send them home (flag 3); wait until this device's own swapped-out outputs are back.
// Fabric counters (GlobalSemaphores) are consumed by an atomic decrement, never reset: an increment of the next call
// that lands early is kept.
void kernel_main() {
    constexpr uint32_t units = get_named_compile_time_arg_val("balance_units");
    constexpr uint32_t cores = get_named_compile_time_arg_val("balance_cores");
    constexpr uint32_t q_blocks = get_named_compile_time_arg_val("balance_q_blocks");
    constexpr uint32_t heads = units / q_blocks;
    constexpr uint32_t passes = get_named_compile_time_arg_val("num_q_passes");
    constexpr uint32_t cap = get_named_compile_time_arg_val("balance_cap");
    constexpr uint32_t cost_words = get_named_compile_time_arg_val("balance_cost_words");
    constexpr bool remote = get_named_compile_time_arg_val("balance_remote") == 1;
    constexpr uint32_t devices = remote ? get_named_compile_time_arg_val("balance_devices") : 1;
    constexpr uint32_t swappable = get_named_compile_time_arg_val("balance_swappable");
    constexpr uint32_t grid_x = get_named_compile_time_arg_val("balance_grid_x");
    constexpr uint32_t sem_costs = get_named_compile_time_arg_val("balance_sem_costs");
    constexpr uint32_t sem_outputs = get_named_compile_time_arg_val("balance_sem_outputs");
    constexpr uint32_t sem_table = get_named_compile_time_arg_val("balance_sem_table");
    constexpr uint32_t sem_import = get_named_compile_time_arg_val("balance_sem_import");
    constexpr uint32_t writer_flag_sem = get_named_compile_time_arg_val("balance_writer_flag");
    constexpr uint32_t cb_scratch = get_named_compile_time_arg_val("balance_cb");
    constexpr uint32_t cb_table = get_named_compile_time_arg_val("cb_balance");
    static_assert(units % q_blocks == 0 && q_blocks <= sparse_balance::kMaxUnitsPerHead, "whole heads, <= 1024 units");
    static_assert(cap <= sparse_balance::kMaxCoreUnits, "a core's units fit its table");
    constexpr uint32_t active = units < cores ? units : cores;
    constexpr uint32_t page_words = cores * cost_words;
    constexpr uint32_t q_blocks_pad = (q_blocks + 15) / 16 * 16;
    constexpr auto table_args = TensorAccessorArgs<0>();
    constexpr auto partner_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t partner_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t me = get_arg_val<uint32_t>(argidx++);  // this device's ring (tensor) rank
    const uint32_t tables_arrived = get_arg_val<uint32_t>(argidx++);
    const uint32_t imports_arrived = get_arg_val<uint32_t>(argidx++);
    const uint32_t returns_arrived = get_arg_val<uint32_t>(argidx++);
    const uint32_t n_writers = get_arg_val<uint32_t>(argidx++);
    const uint32_t xs = argidx;  // NoC x of each SDPA grid column, NoC y of each row, then (x, y) of each gather writer
    const uint32_t ys = xs + grid_x;
    const uint32_t writers = ys + cores / grid_x;
    const auto core_x = [&](uint32_t i) { return get_arg_val<uint32_t>(xs + i % grid_x); };
    const auto core_y = [&](uint32_t i) { return get_arg_val<uint32_t>(ys + i / grid_x); };

    Noc noc;
    const auto table = TensorAccessor(table_args, table_addr);
    const sparse_balance::StaticSplit split{units, cores, q_blocks};
    const sparse_balance::Scratch layout(devices, cores, cost_words, q_blocks, swappable, units);
    CircularBuffer cb(cb_scratch);
    cb.reserve_back(1);
    // The factory sizes the CB one 64-byte line above the layout, so the regions can start 64-byte aligned.
    uint32_t* const base = reinterpret_cast<uint32_t*>((cb.get_write_ptr() + 63) & ~63u);
    ASSERT((layout.words + 16) * sizeof(uint32_t) <= get_local_cb_interface(cb_scratch).fifo_size);
    uint32_t* const tables = base + layout.tables;

    // The gather writers' flag: 1 = this device's cost page is complete, 2 = its partner page is, 3 = the outputs of
    // the row blocks it computes for other devices are written.
    const auto signal_writers = [&]() {
        for (uint32_t w = 0; w < n_writers; ++w) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(writers + 2 * w),
                    get_arg_val<uint32_t>(writers + 2 * w + 1),
                    get_semaphore(writer_flag_sem)),
                1);
        }
        noc_async_atomic_barrier();
    };
    // Wait for a fabric counter on this core to reach n, then consume n of it.
    const auto take_global = [&](uint32_t addr, uint32_t n) {
        volatile tt_l1_ptr uint32_t* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        do {
            invalidate_l1_cache();
        } while (*counter < n);
        noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], addr), 0u - n);
        noc_async_atomic_barrier();
    };

    // 1. Every working SDPA core has written its slice of this device's cost page (and barriered it).
    {
        DeviceZoneScopedN("B-COSTS");  // PROFILE (local only): wait for every SDPA core's costs
        Semaphore<>(sem_costs).wait_min(active);
    }

    // 2. Every device's page, then the swaps.
    uint32_t* const partners = base + layout.partners;
    uint32_t swaps_here = 0;
    {
        DeviceZoneScopedN("B-EXCHANGE");  // PROFILE (local only): the cost pages of every device, read into L1
        if constexpr (remote) {
            signal_writers();
            take_global(tables_arrived, devices - 1);
        }
        for (uint32_t r = 0; r < devices; ++r) {
            noc.async_read(
                table,
                CoreLocalMem<uint32_t>(reinterpret_cast<uint32_t>(tables + r * page_words)),
                page_words * sizeof(uint32_t),
                {.page_id = remote ? r : me},
                {});
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
    }  // PROFILE B-EXCHANGE
    {
        DeviceZoneScopedN("B-ASSIGN");  // PROFILE (local only): swaps, unit tables, releases
        // Cost of device r's flat unit f (r = 0 is this device without cross-device balancing).
        [[maybe_unused]] const auto cost = [&](uint32_t r, uint32_t f) -> uint32_t {
            const uint32_t c = split.core_of(f);
            return tables[r * page_words + c * cost_words + (f - split.start(c))];
        };
        const uint32_t mine = remote ? me : 0u;
        if constexpr (remote) {
            uint32_t* const row_costs = base + layout.row_costs;
            uint32_t* const loads = base + layout.loads;
            // Row-block costs in one sequential walk per page: the static split holds flat units in core order, core
            // c's at its slice.
            for (uint32_t r = 0; r < devices; ++r) {
                uint32_t* const row = row_costs + r * q_blocks;
                for (uint32_t b = 0; b < q_blocks; ++b) {
                    row[b] = 0;
                }
                const uint32_t* const page = tables + r * page_words;
                uint32_t b = 0;  // flat unit % q_blocks
                for (uint32_t c = 0; c < cores; ++c) {
                    const uint32_t* const slice = page + c * cost_words;
                    const uint32_t n = split.count(c);
                    for (uint32_t j = 0; j < n; ++j) {
                        row[b] += slice[j];
                        b = b + 1 == q_blocks ? 0 : b + 1;
                    }
                }
                uint32_t load = 0;
                for (uint32_t block = 0; block < q_blocks; ++block) {
                    load += row[block];
                }
                loads[r] = load;
            }
            sparse_balance::place_swaps(
                devices,
                q_blocks,
                swappable,
                row_costs,
                loads,
                partners,
                reinterpret_cast<uint16_t*>(base + layout.order),
                base + layout.heads,
                base + layout.tails,
                reinterpret_cast<uint8_t*>(base + layout.moved));
            volatile uint32_t* const partner_page = base + layout.partner_page;
            for (uint32_t b = 0; b < q_blocks_pad; ++b) {
                partner_page[b] = b < q_blocks ? partners[me * q_blocks + b] : sparse_balance::kHome;
                swaps_here += b < q_blocks && partner_page[b] != sparse_balance::kHome ? 1u : 0u;
            }
            (void)partner_page[q_blocks_pad - 1];  // the RISC stores land before the NoC reads them
            noc.async_write(
                CoreLocalMem<uint32_t>(reinterpret_cast<uint32_t>(partner_page)),
                TensorAccessor(partner_args, partner_addr),
                q_blocks_pad * sizeof(uint32_t),
                {},
                {.page_id = 0});
            noc.async_write_barrier();
            signal_writers();
        }

        // 3. Unit tables on the costs of the units the slots hold now: the same sequential walk over this device's
        // page, a swapped slot taking its partner unit's cost.
        uint32_t* const unit_costs = base + layout.unit_costs;
        {
            const uint32_t* const page = tables + mine * page_words;
            uint32_t f = 0;
            uint32_t b = 0;  // f % q_blocks
            for (uint32_t c = 0; c < cores; ++c) {
                const uint32_t* const slice = page + c * cost_words;
                const uint32_t n = split.count(c);
                for (uint32_t j = 0; j < n; ++j, ++f) {
                    uint32_t u = slice[j];
                    if constexpr (remote) {
                        const uint32_t p = partners[me * q_blocks + b];
                        if (p != sparse_balance::kHome) {
                            u = cost(p >> 16, f - b + (p & 0xFFFFu));
                        }
                    }
                    unit_costs[f] = u;
                    b = b + 1 == q_blocks ? 0 : b + 1;
                }
            }
        }
        uint32_t* const unit_tables = base + layout.unit_tables;
        {
            DeviceZoneScopedN("B-TABLES");  // PROFILE (local only): plan_unit_tables
            sparse_balance::plan_unit_tables(
                split,
                passes,
                cap,
                unit_costs,
                unit_tables,
                base + layout.core_loads,
                base + layout.bins,
                reinterpret_cast<uint16_t*>(base + layout.unit_order),
                reinterpret_cast<uint16_t*>(base + layout.bin_of));
        }
        // Each core's table (swapped-in units flagged) into its cb_balance page, at the address the CB has on every
        // core of the program (this one included), then the release each core waits on.
        const uint32_t unit_table_addr = CircularBuffer(cb_table).get_write_ptr();
        for (uint32_t i = 0; i < cores; ++i) {
            volatile uint32_t* const unit_table = unit_tables + i * sparse_balance::kTableWords;
            const uint32_t n = unit_table[0];
            if constexpr (remote) {
                for (uint32_t k = 1; k <= n; ++k) {
                    const bool imported = partners[me * q_blocks + unit_table[k] % q_blocks] != sparse_balance::kHome;
                    unit_table[k] = unit_table[k] | (imported ? sparse_balance::kImported : 0u);
                }
            }
            (void)unit_table[n];  // the RISC stores land before the NoC reads them
            noc_async_write(
                reinterpret_cast<uint32_t>(unit_table),
                get_noc_addr(core_x(i), core_y(i), unit_table_addr),
                (n + 1) * sizeof(uint32_t));
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < cores; ++i) {
            noc_inline_dw_write(get_noc_addr(core_x(i), core_y(i), get_semaphore(sem_table)), 1);
        }
        noc_async_write_barrier();
    }  // PROFILE B-ASSIGN

    // 4. Imports in, outputs out, this device's outputs back.
    if constexpr (remote) {
        if (swaps_here > 0) {
            {
                DeviceZoneScopedN("B-IMPORTS");  // PROFILE (local only): wait for the swapped-in blocks
                take_global(imports_arrived, swaps_here);
            }
            for (uint32_t i = 0; i < cores; ++i) {
                noc_semaphore_inc(get_noc_addr(core_x(i), core_y(i), get_semaphore(sem_import)), 1);
            }
            noc_async_atomic_barrier();
            DeviceZoneScopedN("B-OUTPUTS");  // PROFILE (local only): wait for their outputs
            Semaphore<>(sem_outputs).wait_min(swaps_here * heads);
        }
        signal_writers();
        if (swaps_here > 0) {
            DeviceZoneScopedN("B-RETURNS");  // PROFILE (local only): wait for this device's outputs to come back
            take_global(returns_arrived, swaps_here);
        }
    }
}
