// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj writer RISC: the w channel plus store_wave.
// deliver_w_round, w step s = (wave s / num_k_rounds, K round s % num_k_rounds) over the non-empty waves: the
//   n-line's rotating sender multicasts the slot range of the wave's experts (each a k_block_tiles x core_n_tiles
//   chunk in [k][n] order at slot e) from its private staging buffer into the round slots along the line, in one
//   transfer (wa: rect lines, wb: the L strip line). After a send the sender starts the DRAM reads of its next
//   owned step into staging, a quota per step, so every line keeps several steps of w reads in flight. The whole
//   NUM_EXPERTS-slot round is pushed at once.
// store_wave: per tile row m of this core's rows, one untilized block of core_n_tiles tiles -> 32 row segments of y,
//   row 32 * m + i at byte offset line * seg_bytes; one flush per block before the pop.
// Wave p's store runs after wave p + 1's first w_depth w rounds, so compute starts wave p + 1 while the store
// drains (more would deadlock: those slots free only once compute has finished untilizing wave p).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "moe_down_proj_plan.h"

void kernel_main() {
    constexpr uint32_t cb_output_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_counts_writer = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w_round = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w_stage = get_compile_time_arg_val(14);
    constexpr uint32_t prefetch_issue_steps = get_compile_time_arg_val(15);
    constexpr uint32_t tensor_row_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_waves = get_compile_time_arg_val(4);
    constexpr uint32_t num_m_groups = get_compile_time_arg_val(5);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t num_k_rounds = get_compile_time_arg_val(7);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t w_row_pages = get_compile_time_arg_val(10);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t w_depth = get_compile_time_arg_val(13);
    constexpr auto out_args = TensorAccessorArgs<16>();
    constexpr auto w_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto wa_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wa_ct_offset"), get_named_compile_time_arg_val("wa_rt_offset")>();
    constexpr auto wb_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wb_ct_offset"), get_named_compile_time_arg_val("wb_rt_offset")>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t group = get_arg_val<uint32_t>(2);
    const uint32_t line = get_arg_val<uint32_t>(3);

    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t seg_bytes = core_n_tiles * tile_rows * sizeof(uint16_t);
    constexpr uint32_t w_chunk_tiles = k_block_tiles * core_n_tiles;
    constexpr uint32_t w_chunk_bytes = w_chunk_tiles * w_tile_bytes;
    constexpr uint32_t w_round_tiles = MOE_NUM_EXPERTS * w_chunk_tiles;
    constexpr uint32_t w_prefetch_rounds = w_depth < num_k_rounds ? w_depth : num_k_rounds;
    const uint32_t seg_offset = line * seg_bytes;
    const uint32_t n_first = line * core_n_tiles;

    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_bytes);
    const auto w_accessor = TensorAccessor(w_args, w_addr, w_tile_bytes);

    cb_wait_front(cb_counts_writer, 1);
    uint32_t counts[MOE_NUM_EXPERTS];
    {
        volatile tt_l1_ptr uint32_t* landed =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_counts_writer));
        for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
            counts[e] = landed[e];
        }
    }
    const MoePlan plan(counts, tensor_row_tiles, num_waves, num_m_groups);

    Noc noc;
    auto wa_sender = wa_mcast.optional_sender(noc);
    auto wa_receiver = wa_mcast.optional_receiver(noc);
    auto wb_sender = wb_mcast.optional_sender(noc);
    auto wb_receiver = wb_mcast.optional_receiver(noc);
    // Every core sits in exactly one n-line: a wa line (rect) or the wb strip line.
    const bool on_wa = wa_mcast.can_send() || wa_mcast.can_receive();

    auto store_wave = [&](uint32_t wave) {
        const uint32_t row_lo = plan.group_first(wave, group);
        const uint32_t my_rows = plan.group_rows(wave, group);
        for (uint32_t r = 0; r < my_rows; ++r) {
            cb_wait_front(cb_output_rows, core_n_tiles);
            const uint32_t block_l1 = get_read_ptr(cb_output_rows);
            const uint32_t first_row = (row_lo + r) * tile_rows;
            for (uint32_t i = 0; i < tile_rows; ++i) {
                noc_async_write(
                    block_l1 + i * seg_bytes, out_accessor.get_noc_addr(first_row + i, seg_offset), seg_bytes);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_output_rows, core_n_tiles);
        }
    };

    uint32_t num_waves_active = 0;
    while (num_waves_active < num_waves && !plan.wave_empty(num_waves_active)) {
        ++num_waves_active;
    }
    const uint32_t num_steps = num_waves_active * num_k_rounds;
    auto owns = [&](uint32_t step) { return on_wa ? wa_mcast.should_send(step) : wb_mcast.should_send(step); };
    // Writer-private scratch: no CB credits move.
    const uint32_t stage_l1 = get_write_ptr(cb_w_stage);

    // Staged prefetch of this core's next owned step, issued a quota per step so the sender keeps serving the
    // channel rounds in between. Tile t of a step: expert pf_experts[t / w_chunk_tiles], then n, then kk.
    constexpr uint32_t issue_quota = (w_round_tiles + prefetch_issue_steps - 1) / prefetch_issue_steps;
    uint32_t pf_experts[MOE_NUM_EXPERTS];
    uint32_t pf_k_first = 0;
    uint32_t pf_next = 0;
    uint32_t pf_total = 0;
    auto start_prefetch = [&](uint32_t step) {
        pf_next = 0;
        pf_total = 0;
        while (step < num_steps && !owns(step)) {
            ++step;
        }
        if (step >= num_steps) {
            return;
        }
        const uint32_t wave = step / num_k_rounds;
        pf_k_first = (step % num_k_rounds) * k_block_tiles;
        for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
            uint32_t row;
            if (plan.expert_rows_in(e, plan.wave_start(wave), plan.wave_end(wave), &row) != 0) {
                pf_experts[pf_total / w_chunk_tiles] = e;
                pf_total += w_chunk_tiles;
            }
        }
    };
    auto issue_prefetch = [&](uint32_t max_tiles) {
        for (; pf_next < pf_total && max_tiles > 0; ++pf_next, --max_tiles) {
            const uint32_t e = pf_experts[pf_next / w_chunk_tiles];
            const uint32_t n = (pf_next / k_block_tiles) % core_n_tiles;
            const uint32_t kk = pf_next % k_block_tiles;
            const uint32_t page = (n_first + n) * w_row_pages + e * tensor_k_tiles + pf_k_first + kk;
            noc_async_read_page(
                page, w_accessor, stage_l1 + e * w_chunk_bytes + (kk * core_n_tiles + n) * w_tile_bytes);
        }
    };

    start_prefetch(0);
    for (uint32_t wave = 0; wave < num_waves_active; ++wave) {
        uint32_t first_expert;
        uint32_t last_expert;
        plan.wave_experts(wave, &first_expert, &last_expert);
        const uint32_t range_offset = first_expert * w_chunk_bytes;
        const uint32_t range_bytes = (last_expert - first_expert + 1) * w_chunk_bytes;

        for (uint32_t round = 0; round < num_k_rounds; ++round) {
            if (wave > 0 && round == w_prefetch_rounds) {
                store_wave(wave - 1);
            }
            // deliver_w_round
            const uint32_t step = wave * num_k_rounds + round;
            const bool owned = owns(step);
            issue_prefetch(owned ? pf_total : issue_quota);
            cb_reserve_back(cb_w_round, w_round_tiles);
            const uint32_t slot = get_write_ptr(cb_w_round) + range_offset;
            if (owned) {
                noc_async_read_barrier();
                if (on_wa) {
                    wa_sender->send(stage_l1 + range_offset, slot, range_bytes);
                } else {
                    wb_sender->send(stage_l1 + range_offset, slot, range_bytes);
                }
                start_prefetch(step + 1);
            } else if (on_wa) {
                wa_receiver->receive(step);
            } else {
                wb_receiver->receive(step);
            }
            cb_push_back(cb_w_round, w_round_tiles);
        }
        if (wave > 0 && num_k_rounds <= w_prefetch_rounds) {
            store_wave(wave - 1);
        }
    }
    if (num_waves_active > 0) {
        store_wave(num_waves_active - 1);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_counts_writer, 1);
}
