// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Port reader (NCRISC, NoC0): per block it sends, per chunk: my partial (input) and, for a relayed block, what
// upstream sent (scratch, once the arrival counter covers it). The first `lead` blocks start their chain here (no
// upstream): my partial only, into cb::OWN for the compute's pass-through, or straight to the sender's cb::OUT when
// every block leads (a line end).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/fabric_reduce_scatter/device/kernels/fabric_reduce_scatter_args.hpp"
#include "ttnn/cpp/ttnn/operations/fabric_reduce_scatter/device/kernels/fabric_reduce_scatter_chunk_walk.hpp"

using namespace fabric_reduce_scatter;

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(ct::PAGE_BYTES);
    constexpr uint32_t run_pages = get_compile_time_arg_val(ct::RUN_PAGES);
    constexpr uint32_t num_banks = get_compile_time_arg_val(ct::NUM_BANKS);
    constexpr uint32_t group = get_compile_time_arg_val(ct::GROUP);
    constexpr uint32_t inc_every = get_compile_time_arg_val(ct::INC_EVERY);
    [[maybe_unused]] constexpr uint32_t col_wt = get_compile_time_arg_val(read_ct::COL_WT);
    constexpr uint32_t col_wb = get_compile_time_arg_val(read_ct::COL_WB);
    [[maybe_unused]] constexpr uint32_t col_div = col_wb == 0 ? 1 : col_wb;
    constexpr auto in_args = TensorAccessorArgs<read_ct::COUNT>();
    constexpr auto scr_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(port_reader_common::IN_ADDR);
    const uint32_t scr_addr = get_common_arg_val<uint32_t>(port_reader_common::SCR_ADDR);
    const uint32_t blk_pages = get_common_arg_val<uint32_t>(port_reader_common::BLK_PAGES);
    const uint32_t arrival_addr = get_common_arg_val<uint32_t>(port_reader_common::ARRIVAL_ADDR);
    const uint32_t first_bank = get_arg_val<uint32_t>(port_reader_rt::FIRST_BANK);
    const uint32_t bank_stride = get_arg_val<uint32_t>(port_reader_rt::BANK_STRIDE);
    const uint32_t full = get_arg_val<uint32_t>(port_reader_rt::FULL);  // chunks of one block for this port
    const uint32_t lead = get_arg_val<uint32_t>(port_reader_rt::LEAD);  // leading blocks without upstream
    const uint32_t nblocks = get_arg_val<uint32_t>(port_reader_rt::NUM_BLOCKS);
    constexpr uint32_t blocks_idx = port_reader_rt::COUNT;
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    // Input pages of block `blk`, block-local `page`, n pages consecutive in their bank. Row blocks (col_wb == 0):
    // one read. Column blocks: block-local page q is input page (q / col_wb) col_wt + blk col_wb + q % col_wb, each
    // read on its own.
    auto read_in = [&](uint32_t blk, uint32_t page, uint32_t n, uint32_t dst) {
        if constexpr (col_wb == 0) {
            noc_async_read(in.get_noc_addr(blk * blk_pages + page), dst, n * page_bytes);
        } else {
            for (uint32_t t = 0; t < n; ++t) {
                const uint32_t q = page + t * num_banks;
                noc_async_read(
                    in.get_noc_addr((q / col_div) * col_wt + blk * col_wb + q % col_div),
                    dst + t * page_bytes,
                    page_bytes);
            }
        }
    };

    // A batch is written contiguously from the CB's write pointer, so it must not cross the CB's end: the CBs hold
    // cap = 2 group chunks, and with two phases (or OWN and IN starting at different offsets) the batches are not
    // group-aligned. o0 / o1: the chunk offset of the next write in the own-partial CB (OWN, or OUT) and in IN.
    constexpr uint32_t cap = 2 * group;
    uint32_t o0 = 0, o1 = 0;
    auto room = [&](uint32_t o) { return cap - o < group ? cap - o : group; };

    // leading blocks: my partial only
    const uint32_t cb_lead = lead == nblocks ? cb::OUT : cb::OWN;
    uint32_t chunks_left = lead * full;
    uint32_t batch = 0, limit = 0, wown = 0, win = 0;
    for (uint32_t k = 0; k < lead; ++k) {
        const uint32_t blk = get_arg_val<uint32_t>(blocks_idx + k);
        for_each_chunk(
            blk_pages, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n, uint32_t) {
                if (batch == 0) {
                    limit = room(o0);
                    cb_reserve_back(cb_lead, run_pages * limit);
                    wown = get_write_ptr(cb_lead);
                }
                read_in(blk, page, n, wown + batch * chunk_bytes);
                --chunks_left;
                if (++batch == limit || chunks_left == 0) {
                    noc_async_read_barrier();
                    cb_push_back(cb_lead, run_pages * batch);
                    o0 = (o0 + batch) % cap;
                    batch = 0;
                }
            });
    }
    if (cb_lead != cb::OWN) {
        o0 = 0;  // the relay phase's OWN is untouched
    }
    // relayed blocks: my partial (OWN) and upstream's running sum (IN); upstream's relay stream starts at my block lead
    chunks_left = (nblocks - lead) * full;
    for (uint32_t k = lead; k < nblocks; ++k) {
        const uint32_t blk = get_arg_val<uint32_t>(blocks_idx + k);
        const uint32_t base = blk * blk_pages;
        for_each_chunk(
            blk_pages, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n, uint32_t idx) {
                if (batch == 0) {
                    limit = room(o0) < room(o1) ? room(o0) : room(o1);
                    cb_reserve_back(cb::OWN, run_pages * limit);
                    wown = get_write_ptr(cb::OWN);
                    cb_reserve_back(cb::IN, run_pages * limit);
                    win = get_write_ptr(cb::IN);
                }
                read_in(blk, page, n, wown + batch * chunk_bytes);
                noc_semaphore_wait_min(arrived, ((k - lead) * full + idx) / inc_every + 1);
                noc_async_read(scr.get_noc_addr(base + page), win + batch * chunk_bytes, n * page_bytes);
                --chunks_left;
                if (++batch == limit || chunks_left == 0) {
                    noc_async_read_barrier();
                    cb_push_back(cb::OWN, run_pages * batch);
                    cb_push_back(cb::IN, run_pages * batch);
                    o0 = (o0 + batch) % cap;
                    o1 = (o1 + batch) % cap;
                    batch = 0;
                }
            });
    }
}
