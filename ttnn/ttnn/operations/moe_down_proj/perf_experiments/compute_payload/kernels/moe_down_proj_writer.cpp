// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj writer RISC: the w channel plus store_wave (moe_down_proj_store.h) unless store_on_reader.
// deliver_w_round, w step s = (wave s / num_k_rounds, K round s % num_k_rounds) over the non-empty waves: the
//   n-line's rotating sender multicasts the slot range of the wave's experts (each a k_block_tiles x core_n_tiles
//   chunk in [k][n] order at slot e) from its private staging buffer into the round slots along the line, in one
//   transfer (wa: rect lines, wb: the L strip line). The staged DRAM reads (moe_down_proj_w_prefetch.h) run here
//   after each send, a quota per step, unless w_reads_on_reader: then the reader RISC fills cb_w_stage on NoC1 and
//   this RISC waits for it, so the read responses do not share NoC0 with the w multicast. The whole
//   NUM_EXPERTS-slot round is pushed at once.
// Wave p's store starts after wave p + 1's first w_depth w rounds, so compute starts wave p + 1 while the store
// drains, store_blocks_per_round blocks per round (the host sizes cb_output_rows to the whole wave block when it
// paces, so compute's untilize never waits on the store; unpaced, the quota covers the whole wave).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "moe_down_proj_plan.h"
#include "moe_down_proj_store.h"
#include "moe_down_proj_w_prefetch.h"

void kernel_main() {
    constexpr uint32_t cb_output_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_counts_writer = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w_round = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_row_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_waves = get_compile_time_arg_val(4);
    constexpr uint32_t num_m_groups = get_compile_time_arg_val(5);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t num_k_rounds = get_compile_time_arg_val(7);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t w_reader_share = get_compile_time_arg_val(10);
    constexpr bool w_reads_on_reader = w_reader_share > 0;
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t w_depth = get_compile_time_arg_val(13);
    constexpr uint32_t cb_w_stage = get_compile_time_arg_val(14);
    // Staged w DRAM reads issued per channel step (host: w_issue_quota()).
    constexpr uint32_t issue_quota = get_compile_time_arg_val(15);
    // The reader RISC (the other NoC) owns store_wave instead.
    constexpr bool store_on_reader = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t store_blocks_per_round = get_compile_time_arg_val(17);
    constexpr uint32_t cb_w_owner = get_compile_time_arg_val(18);
    constexpr uint32_t w_share_den = get_compile_time_arg_val(19);
    constexpr auto out_args = TensorAccessorArgs<20>();
    constexpr auto w_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto wa_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wa_ct_offset"), get_named_compile_time_arg_val("wa_rt_offset")>();
    constexpr auto wb_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wb_ct_offset"), get_named_compile_time_arg_val("wb_rt_offset")>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t group = get_arg_val<uint32_t>(2);
    const uint32_t line = get_arg_val<uint32_t>(3);

    constexpr uint32_t w_chunk_bytes = k_block_tiles * core_n_tiles * w_tile_bytes;
    constexpr uint32_t w_round_tiles = MOE_NUM_EXPERTS * k_block_tiles * core_n_tiles;
    constexpr uint32_t w_prefetch_rounds = w_depth < num_k_rounds ? w_depth : num_k_rounds;
    const uint32_t n_first = line * core_n_tiles;

    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_bytes);
    const auto w_accessor = TensorAccessor(w_args, w_addr, w_tile_bytes);

    Noc noc;
    auto wa_sender = wa_mcast.optional_sender(noc);
    auto wa_receiver = wa_mcast.optional_receiver(noc);
    auto wb_sender = wb_mcast.optional_sender(noc);
    auto wb_receiver = wb_mcast.optional_receiver(noc);
    // Every core sits in exactly one n-line: a wa line (rect) or the wb strip line.
    const bool on_wa = wa_mcast.can_send() || wa_mcast.can_receive();
    auto owns = [&](uint32_t step) { return on_wa ? wa_mcast.should_send(step) : wb_mcast.should_send(step); };
    if constexpr (w_reads_on_reader) {
        moe_publish_w_owner<cb_w_owner>(owns);
    }

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

    auto store_wave = [&](uint32_t wave, uint32_t first = 0, uint32_t max_blocks = UINT32_MAX) {
        if constexpr (!store_on_reader) {
            return moe_store_wave<cb_output_rows, core_n_tiles>(
                plan, out_accessor, wave, group, line, first, max_blocks);
        }
        return first;
    };

    uint32_t num_waves_active = 0;
    while (num_waves_active < num_waves && !plan.wave_empty(num_waves_active)) {
        ++num_waves_active;
    }
    const uint32_t num_steps = num_waves_active * num_k_rounds;
    // Writer-private scratch unless w_reads_on_reader (then the reader produces it and this RISC consumes it).
    const uint32_t stage_l1 = get_write_ptr(cb_w_stage);
    MoeWPrefetch<
        k_block_tiles,
        core_n_tiles,
        tensor_k_tiles,
        num_k_rounds,
        w_tile_bytes,
        w_share_den,
        decltype(w_accessor)>
        prefetch(plan, w_accessor, n_first, w_reader_share, w_share_den);

    prefetch.start(0, num_steps, owns);
    for (uint32_t wave = 0; wave < num_waves_active; ++wave) {
        uint32_t first_expert;
        uint32_t last_expert;
        plan.wave_experts(wave, &first_expert, &last_expert);
        const uint32_t range_offset = first_expert * w_chunk_bytes;
#ifdef ABL_NO_W
        const uint32_t range_bytes = w_tile_bytes;
#else
        const uint32_t range_bytes = (last_expert - first_expert + 1) * w_chunk_bytes;
#endif

        uint32_t stored = 0;
        for (uint32_t round = 0; round < num_k_rounds; ++round) {
            if (wave > 0 && round >= w_prefetch_rounds) {
                stored = store_wave(wave - 1, stored, store_blocks_per_round);
            }
            // deliver_w_round
            const uint32_t step = wave * num_k_rounds + round;
            const bool owned = owns(step);
#ifndef ABL_NO_W
            prefetch.issue(stage_l1, owned ? prefetch.total : issue_quota);
#endif
            {
                MaybeDeviceZoneScope("w_reserve");
                cb_reserve_back(cb_w_round, w_round_tiles);
            }
            const uint32_t slot = get_write_ptr(cb_w_round) + range_offset;
            if (owned) {
                MaybeDeviceZoneScope("w_send");
                {
                    MaybeDeviceZoneScope("w_send_barrier");
                    // The single-round stage CB keeps its read and write pointers at stage_l1.
                    if constexpr (w_reads_on_reader) {
                        cb_wait_front(cb_w_stage, w_round_tiles);
                    }
                    noc_async_read_barrier();
                }
                if (on_wa) {
                    wa_sender->send(stage_l1 + range_offset, slot, range_bytes);
                } else {
                    wb_sender->send(stage_l1 + range_offset, slot, range_bytes);
                }
                if constexpr (w_reads_on_reader) {
                    cb_pop_front(cb_w_stage, w_round_tiles);
                }
                prefetch.start(step + 1, num_steps, owns);
            } else if (on_wa) {
                MaybeDeviceZoneScope("w_receive");
                wa_receiver->receive(step);
            } else {
                MaybeDeviceZoneScope("w_receive");
                wb_receiver->receive(step);
            }
            cb_push_back(cb_w_round, w_round_tiles);
        }
        if (wave > 0) {
            store_wave(wave - 1, stored);
        }
    }
    if (num_waves_active > 0) {
        store_wave(num_waves_active - 1);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_counts_writer, 1);
}
