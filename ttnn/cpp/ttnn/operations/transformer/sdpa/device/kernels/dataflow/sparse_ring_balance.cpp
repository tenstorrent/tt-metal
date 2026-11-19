// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_ring_balance_plan.hpp"

// Load balancing of sparse_ring_sdpa, one core per device, every call:
//   1. wait for every SDPA core's unit costs in this device's table page (DRAM);
//   2. cross-device: share the pages through the gather writers and compute the row-block swaps (sparse_balance::
//      place_swaps, identically on every device), then have the writers send the swapped row blocks;
//   3. plan every core's unit table (sparse_balance::plan_unit_tables) and release it to the SDPA cores;
//   4. cross-device: signal the SDPA cores once the imports arrive, have the writers send their outputs home, and wait
//      for this device's own outputs to come back.
// Fabric counters are consumed by an atomic decrement, never reset, so an early increment of the next call is kept.
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
    constexpr uint32_t row_offset = cores * cost_words;  // a table page's row-block costs
    constexpr uint32_t page_words = sparse_balance::table_page_words(cores, cost_words, q_blocks);
    constexpr uint32_t q_blocks_pad = (q_blocks + 15) / 16 * 16;
    constexpr sparse_balance::StaticSplit split{units, cores, q_blocks};
    constexpr uint32_t max_bins = split.max_bins();
    static_assert(max_bins <= sparse_balance::kMaxBins, "a head's bins fit a byte");
    using Local = sparse_balance::LocalScratch<q_blocks, max_bins>;
    static_assert(
        sizeof(Local) <= sparse_balance::local_scratch_bytes(q_blocks, max_bins) &&
            sparse_balance::local_scratch_bytes(q_blocks, max_bins) <= sparse_balance::kMaxLocalScratchBytes,
        "the local scratch fits its budget");
    static Local local;
    constexpr auto table_args = TensorAccessorArgs<0>();
    constexpr auto partner_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t partner_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t me = get_arg_val<uint32_t>(argidx++);  // this device's ring (tensor) rank
    const uint32_t tables_arrived = get_arg_val<uint32_t>(argidx++);
    const uint32_t imports_arrived = get_arg_val<uint32_t>(argidx++);
    const uint32_t returns_arrived = get_arg_val<uint32_t>(argidx++);
    const uint32_t tables_read = get_arg_val<uint32_t>(argidx++);
    const uint32_t n_writers = get_arg_val<uint32_t>(argidx++);
    const uint32_t xs = argidx;  // NoC x of each SDPA grid column, NoC y of each row, then (x, y) of each gather writer
    const uint32_t ys = xs + grid_x;
    const uint32_t writers = ys + cores / grid_x;
    const auto core_x = [&](uint32_t i) { return get_arg_val<uint32_t>(xs + i % grid_x); };
    const auto core_y = [&](uint32_t i) { return get_arg_val<uint32_t>(ys + i / grid_x); };

    Noc noc;
    const auto table = TensorAccessor(table_args, table_addr);
    const sparse_balance::Scratch layout(devices, cores, cost_words, q_blocks, swappable, units);
    CircularBuffer cb(cb_scratch);
    cb.reserve_back(1);
    // The factory sizes the CB 64 bytes above the layout for this alignment.
    uint32_t* const base = reinterpret_cast<uint32_t*>((cb.get_write_ptr() + 63) & ~63u);
    ASSERT((layout.words + 16) * sizeof(uint32_t) <= get_local_cb_interface(cb_scratch).fifo_size);
    uint32_t* const tables = base + layout.tables;

    // The gather writers' flag: 1 = the cost page is ready, 2 = the partner page is, 3 = the imports' outputs are.
    const auto signal_writers = [&]() {
        for (uint32_t w = 0; w < n_writers; ++w) {
            Semaphore<>(writer_flag_sem)
                .up(noc, get_arg_val<uint32_t>(writers + 2 * w), get_arg_val<uint32_t>(writers + 2 * w + 1), 1);
        }
        noc.async_atomic_barrier();
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

    // 1. Every working SDPA core has written its costs.
    Semaphore<>(sem_costs).wait_min(active);

    // 2. This device's unit costs (in flat unit order) and row-block costs.
    const uint32_t mine = remote ? me : 0u;
    const auto read_page = [&](uint32_t r) {
        noc.async_read(
            table,
            CoreLocalMem<uint32_t>(reinterpret_cast<uint32_t>(tables + r * page_words)),
            page_words * sizeof(uint32_t),
            {.page_id = remote ? r : me},
            {});
    };
    read_page(mine);
    noc.async_read_barrier();
    invalidate_l1_cache();
    uint32_t* const unit_costs = base + layout.unit_costs;
    {
        uint32_t* const row = local.swap.row;
        for (uint32_t b = 0; b < q_blocks; ++b) {
            row[b] = 0;
        }
        const uint32_t* const page = tables + mine * page_words;
        uint32_t f = 0;
        uint32_t b = 0;  // f % q_blocks
        for (uint32_t c = 0; c < cores; ++c) {
            const uint32_t* const slice = page + c * cost_words;
            const uint32_t n = split.count(c);
            for (uint32_t j = 0; j < n; ++j, ++f) {
                const uint32_t u = slice[j];
                unit_costs[f] = u;
                if constexpr (remote) {
                    row[b] += u;
                }
                b = b + 1 == q_blocks ? 0 : b + 1;
            }
        }
    }

    // Cross-device: publish the row-block costs, read every device's page, place the swaps.
    uint32_t* const partners = base + layout.partners;
    uint32_t swaps_here = 0;
    if constexpr (remote) {
        volatile uint32_t* const tail = tables + me * page_words + row_offset;
        for (uint32_t b = 0; b < q_blocks_pad; ++b) {
            tail[b] = b < q_blocks ? local.swap.row[b] : 0u;
        }
        sparse_balance::flush_stores(tail, q_blocks_pad - 1);
        noc.async_write(
            CoreLocalMem<uint32_t>(reinterpret_cast<uint32_t>(tail)),
            table,
            q_blocks_pad * sizeof(uint32_t),
            {},
            {.page_id = me, .offset_bytes = row_offset * sizeof(uint32_t)});
        noc.async_write_barrier();
        signal_writers();
        take_global(tables_arrived, devices - 1);
        for (uint32_t r = 0; r < devices; ++r) {
            if (r != me) {
                read_page(r);
            }
        }
        noc.async_read_barrier();
        invalidate_l1_cache();

        uint32_t* const row_costs = base + layout.row_costs;
        uint32_t* const loads = base + layout.loads;
        uint16_t* const order = reinterpret_cast<uint16_t*>(base + layout.order);
        for (uint32_t r = 0; r < devices; ++r) {
            const uint32_t* const page_rows = tables + r * page_words + row_offset;
            uint32_t load = 0;
            uint32_t top = 0;
            for (uint32_t b = 0; b < q_blocks; ++b) {
                const uint32_t c = page_rows[b];
                local.swap.row[b] = c;
                row_costs[r * q_blocks + b] = c;
                load += c;
                top = b < swappable && c > top ? c : top;
            }
            loads[r] = load;
            sparse_balance::sort_heaviest_first(local.swap.row, swappable, top, local.swap.order, local.swap.tmp);
            for (uint32_t k = 0; k < swappable; ++k) {
                order[r * swappable + k] = local.swap.order[k];
            }
        }
        sparse_balance::place_swaps(
            devices,
            q_blocks,
            swappable,
            row_costs,
            loads,
            partners,
            order,
            base + layout.heads,
            base + layout.tails,
            reinterpret_cast<uint8_t*>(base + layout.moved));
        volatile uint32_t* const partner_page = base + layout.partner_page;
        for (uint32_t b = 0; b < q_blocks_pad; ++b) {
            const uint32_t p = b < q_blocks ? partners[me * q_blocks + b] : sparse_balance::kHome;
            partner_page[b] = p;
            if (b < q_blocks) {
                local.imported[b] = p != sparse_balance::kHome ? 1 : 0;
                swaps_here += local.imported[b];
            }
        }
        sparse_balance::flush_stores(partner_page, q_blocks_pad - 1);
        noc.async_write(
            CoreLocalMem<uint32_t>(reinterpret_cast<uint32_t>(partner_page)),
            TensorAccessor(partner_args, partner_addr),
            q_blocks_pad * sizeof(uint32_t),
            {},
            {.page_id = 0});
        noc.async_write_barrier();
        signal_writers();

        // A swapped slot takes its partner unit's cost.
        for (uint32_t b = 0; b < q_blocks; ++b) {
            if (local.imported[b]) {
                const uint32_t p = partners[me * q_blocks + b];
                const uint32_t* const page = tables + (p >> 16) * page_words;
                for (uint32_t f0 = 0; f0 < units; f0 += q_blocks) {
                    const uint32_t f = f0 + (p & 0xFFFFu);
                    const uint32_t c = split.core_of(f);
                    unit_costs[f0 + b] = page[c * cost_words + (f - split.start(c))];
                }
            }
        }
    }

    // 3. Unit tables on the costs of the units the slots now hold.
    uint32_t* const unit_tables = base + layout.unit_tables;
    sparse_balance::plan_unit_tables(
        split,
        passes,
        cap,
        unit_costs,
        remote ? local.imported : nullptr,
        unit_tables,
        base + layout.core_loads,
        base + layout.ranked,
        local.head());
    // Each core's table into its cb_balance page (the CB has the same address on every core), then its release.
    const uint32_t unit_table_addr = CircularBuffer(cb_table).get_write_ptr();
    for (uint32_t i = 0; i < cores; ++i) {
        volatile uint32_t* const unit_table = unit_tables + i * sparse_balance::kTableWords;
        const uint32_t n = unit_table[0];
        sparse_balance::flush_stores(unit_table, n);
        noc.async_write(
            CoreLocalMem<uint32_t>(reinterpret_cast<uint32_t>(unit_table)),
            UnicastEndpoint{},
            (n + 1) * sizeof(uint32_t),
            {},
            {.noc_x = core_x(i), .noc_y = core_y(i), .addr = unit_table_addr});
    }
    noc.async_write_barrier();
    for (uint32_t i = 0; i < cores; ++i) {
        noc.inline_dw_write(
            UnicastEndpoint{}, 1, {.noc_x = core_x(i), .noc_y = core_y(i), .addr = get_semaphore(sem_table)});
    }
    noc.async_write_barrier();

    // 4. Imports in, outputs out, this device's outputs back.
    if constexpr (remote) {
        if (swaps_here > 0) {
            take_global(imports_arrived, swaps_here);
            for (uint32_t i = 0; i < cores; ++i) {
                Semaphore<>(sem_import).up(noc, core_x(i), core_y(i), 1);
            }
            noc.async_atomic_barrier();
            Semaphore<>(sem_outputs).wait_min(swaps_here * heads);
        }
        signal_writers();
        if (swaps_here > 0) {
            take_global(returns_arrived, swaps_here);
        }
        // Every other device has read this device's page, so the next call may overwrite it.
        take_global(tables_read, devices - 1);
    }
}
