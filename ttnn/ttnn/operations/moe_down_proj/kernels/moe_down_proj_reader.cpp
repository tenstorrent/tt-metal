// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj reader (x channel; the w channel runs on the writer RISC), plus store_wave when store_on_reader.
// D0: counts page -> cb_counts_compute and cb_counts_writer (parsed from the landing copy for the plan).
// deliver_x_round, x step s = (wave s / num_k_rounds, K round s % num_k_rounds) over the non-empty waves:
//   the step's rotating sender of the m-group multicasts its rows' K block of x from its private staging buffer
//   into the round slot along the m-group (xa); in the L layout it also copies the block to the group's strip core
//   (xb). After a send the sender starts the reads of its next owned step into staging, a quota per step, so a
//   step's L1 reads overlap the steps in between. Nominal core_row_tiles_max * k_block_tiles pages pushed per step; an
//   empty m-group moves no data.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "moe_down_proj_plan.h"
#include "moe_down_proj_store.h"

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x_stage = get_compile_time_arg_val(1);
    constexpr uint32_t cb_counts_compute = get_compile_time_arg_val(2);
    constexpr uint32_t cb_counts_writer = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t num_waves = get_compile_time_arg_val(5);
    constexpr uint32_t num_m_groups = get_compile_time_arg_val(6);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t num_k_rounds = get_compile_time_arg_val(8);
    constexpr uint32_t core_row_tiles_max = get_compile_time_arg_val(9);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t counts_page_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t prefetch_issue_steps = get_compile_time_arg_val(13);
    // store_wave runs here (on this RISC's NoC) instead of on the writer RISC.
    constexpr bool store_on_reader = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t cb_output_rows = get_compile_time_arg_val(15);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t x_depth = get_compile_time_arg_val(18);
    constexpr uint32_t store_blocks_per_round = get_compile_time_arg_val(19);
    constexpr auto x_args = TensorAccessorArgs<20>();
    constexpr auto counts_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto out_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto xa_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("xa_ct_offset"), get_named_compile_time_arg_val("xa_rt_offset")>();
    constexpr auto xb_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("xb_ct_offset"), get_named_compile_time_arg_val("xb_rt_offset")>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_arg_val<uint32_t>(1);
    const uint32_t group = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t line = get_arg_val<uint32_t>(4);

    constexpr uint32_t x_round_tiles = core_row_tiles_max * k_block_tiles;

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto counts_accessor = TensorAccessor(counts_args, counts_addr, counts_page_bytes);

    // D0: one read per consumer CB (single-consumer rule).
    cb_reserve_back(cb_counts_compute, 1);
    cb_reserve_back(cb_counts_writer, 1);
    const uint32_t counts_l1 = get_write_ptr(cb_counts_compute);
    noc_async_read_page(0, counts_accessor, counts_l1);
    noc_async_read_page(0, counts_accessor, get_write_ptr(cb_counts_writer));
    noc_async_read_barrier();
    uint32_t counts[MOE_NUM_EXPERTS];
    {
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
        for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
            counts[e] = landed[e];
        }
    }
    cb_push_back(cb_counts_compute, 1);
    cb_push_back(cb_counts_writer, 1);

    const MoePlan plan(counts, tensor_row_tiles, num_waves, num_m_groups);
    // Non-empty waves are a prefix (a wave is empty only past the active rows).
    uint32_t num_steps = 0;
    for (uint32_t wave = 0; wave < num_waves && !plan.wave_empty(wave); ++wave) {
        num_steps += num_k_rounds;
    }

    Noc noc;
    auto xa_sender = xa_mcast.optional_sender(noc);
    auto xa_receiver = xa_mcast.optional_receiver(noc);
    auto xb_sender = xb_mcast.optional_sender(noc);
    auto xb_receiver = xb_mcast.optional_receiver(noc);
    // Rect cores own an xa seat (and send xb in the L layout); L strip cores only receive xb.
    const bool on_xa = xa_mcast.can_send() || xa_mcast.can_receive();
    // Reader-private scratch: no CB credits move.
    const uint32_t stage_l1 = get_write_ptr(cb_x_stage);

    // Staged prefetch of this core's next owned step. Its reads are issued a quota per step (rather than all at
    // once after a send), so the sender keeps serving the channel rounds in between.
    constexpr uint32_t issue_quota = (x_round_tiles + prefetch_issue_steps - 1) / prefetch_issue_steps;
    uint32_t pf_first_page = 0;
    uint32_t pf_next = 0;
    uint32_t pf_total = 0;
    auto start_prefetch = [&](uint32_t step) {
        pf_next = 0;
        pf_total = 0;
        if (!on_xa) {
            return;
        }
        while (step < num_steps && !xa_mcast.should_send(step)) {
            ++step;
        }
        if (step >= num_steps) {
            return;
        }
        const uint32_t wave = step / num_k_rounds;
        pf_first_page = plan.group_first(wave, group) * tensor_k_tiles + (step % num_k_rounds) * k_block_tiles;
        pf_total = plan.group_rows(wave, group) * k_block_tiles;
    };
    auto issue_prefetch = [&](uint32_t max_tiles) {
        for (; pf_next < pf_total && max_tiles > 0; ++pf_next, --max_tiles) {
            const uint32_t row = pf_next / k_block_tiles;
            const uint32_t kk = pf_next % k_block_tiles;
            noc_async_read_page(pf_first_page + row * tensor_k_tiles + kk, x_accessor, stage_l1 + pf_next * tile_bytes);
        }
    };

    // Wave p's store starts after wave p + 1's first x_store_round x rounds (the x slots compute can fill before it
    // has untilized wave p), so compute starts wave p + 1 while the store drains, store_blocks_per_round per round.
    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_bytes);
    constexpr uint32_t x_store_round = x_depth < num_k_rounds ? x_depth : num_k_rounds;
    auto store_wave = [&](uint32_t wave, uint32_t first = 0, uint32_t max_blocks = UINT32_MAX) {
        if constexpr (store_on_reader) {
            return moe_store_wave<cb_output_rows, core_n_tiles>(
                plan, out_accessor, wave, group, line, first, max_blocks);
        }
        return first;
    };

    start_prefetch(0);
    uint32_t stored = 0;
    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t round = step % num_k_rounds;
        if (round == 0) {
            stored = 0;
        }
        if (step >= num_k_rounds && round >= x_store_round) {
            stored = store_wave(step / num_k_rounds - 1, stored, store_blocks_per_round);
        }
        const uint32_t group_rows = plan.group_rows(step / num_k_rounds, group);
        const bool owned = on_xa && xa_mcast.should_send(step);
        issue_prefetch(owned ? pf_total : issue_quota);
        cb_reserve_back(cb_x_round, x_round_tiles);
        if (group_rows > 0) {
            const uint32_t slot = get_write_ptr(cb_x_round);
            const uint32_t bytes = group_rows * k_block_tiles * tile_bytes;
            if (owned) {
                noc_async_read_barrier();
                xa_sender->send(stage_l1, slot, bytes);
                if (xb_mcast.can_send()) {
                    xb_sender->send(stage_l1, slot, bytes);
                }
            } else if (on_xa) {
                xa_receiver->receive(step);
            } else {
                xb_receiver->receive(step);
            }
        }
        if (owned) {
            start_prefetch(step + 1);
        }
        cb_push_back(cb_x_round, x_round_tiles);
        if (step >= num_k_rounds && round == num_k_rounds - 1) {
            store_wave(step / num_k_rounds - 1, stored);
        }
    }
    if (num_steps > 0) {
        store_wave(num_steps / num_k_rounds - 1);
    }
    if constexpr (store_on_reader) {
        noc_async_write_barrier();
    }
}
