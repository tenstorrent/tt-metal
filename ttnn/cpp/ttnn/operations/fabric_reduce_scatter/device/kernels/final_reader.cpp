// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Final reader (NCRISC, NoC0): per chunk of my own block (my half of my link's banks): my partial, the forward
// arrival (scratch slot own, counter A) and the backward arrival (scratch slot G, counter B), each only if that
// direction has an upstream.

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

    const uint32_t in_addr = get_common_arg_val<uint32_t>(final_reader_common::IN_ADDR);
    const uint32_t scr_addr = get_common_arg_val<uint32_t>(final_reader_common::SCR_ADDR);
    const uint32_t blk_pages = get_common_arg_val<uint32_t>(final_reader_common::BLK_PAGES);
    const uint32_t bwd_base = get_common_arg_val<uint32_t>(final_reader_common::BWD_SLOT) * blk_pages;
    const uint32_t a_addr = get_common_arg_val<uint32_t>(final_reader_common::SEM_A_ADDR);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(final_reader_common::SEM_B_ADDR);
    const uint32_t first_bank = get_arg_val<uint32_t>(final_reader_rt::FIRST_BANK);
    const uint32_t bank_stride = get_arg_val<uint32_t>(final_reader_rt::BANK_STRIDE);
    const uint32_t own_blk = get_arg_val<uint32_t>(final_reader_rt::OWN_SLOT);
    const uint32_t fwd_base = get_arg_val<uint32_t>(final_reader_rt::FWD_SLOT) * blk_pages;
    const uint32_t has_fwd = get_arg_val<uint32_t>(final_reader_rt::HAS_FWD);
    const uint32_t has_bwd = get_arg_val<uint32_t>(final_reader_rt::HAS_BWD);
    uint32_t chunks_left = get_arg_val<uint32_t>(final_reader_rt::NUM_CHUNKS);
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arr_a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_addr);
    volatile tt_l1_ptr uint32_t* arr_b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_addr);

    // Input pages of my own block (see port_reader.cpp).
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

    uint32_t batch = 0, w0 = 0, w1 = 0, w2 = 0;
    for_each_chunk(
        blk_pages, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n, uint32_t idx) {
            if (batch == 0) {
                cb_reserve_back(cb::OWN, run_pages * group);
                w0 = get_write_ptr(cb::OWN);
                if (has_fwd) {
                    cb_reserve_back(cb::IN, run_pages * group);
                    w1 = get_write_ptr(cb::IN);
                }
                if (has_bwd) {
                    cb_reserve_back(cb::IN2, run_pages * group);
                    w2 = get_write_ptr(cb::IN2);
                }
            }
            read_in(own_blk, page, n, w0 + batch * chunk_bytes);
            if (has_fwd) {
                noc_semaphore_wait_min(arr_a, idx / inc_every + 1);
                noc_async_read(scr.get_noc_addr(fwd_base + page), w1 + batch * chunk_bytes, n * page_bytes);
            }
            if (has_bwd) {
                noc_semaphore_wait_min(arr_b, idx / inc_every + 1);
                noc_async_read(scr.get_noc_addr(bwd_base + page), w2 + batch * chunk_bytes, n * page_bytes);
            }
            --chunks_left;
            if (++batch == group || chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(cb::OWN, run_pages * batch);
                if (has_fwd) {
                    cb_push_back(cb::IN, run_pages * batch);
                }
                if (has_bwd) {
                    cb_push_back(cb::IN2, run_pages * batch);
                }
                batch = 0;
            }
        });
}
