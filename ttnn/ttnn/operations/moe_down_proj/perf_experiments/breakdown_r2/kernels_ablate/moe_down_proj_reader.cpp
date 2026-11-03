// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj reader (x channel; the w channel runs on the writer RISC), plus store_wave when store_on_reader.
// The last wave's store is split with the writer RISC either way (moe_store_tail_owner / _helper).
// D0: counts page -> cb_counts_compute and cb_counts_writer (parsed from the landing copy for the plan).
// deliver_x_round, x step s = (wave s / x_steps_per_wave, x K block s % x_steps_per_wave) over the non-empty waves:
//   the step's rotating sender of the m-group multicasts its rows' K block of x from its private staging buffer
//   into the round slot along the m-group (xa). In the L layout a fixed forwarder of the rect row (never an x
//   sender) copies each step to the group's strip core (xb) one step behind, from its own round slot, before it
//   acks the next step, so the sender injects each step once. After a send the sender starts the reads of its
//   next owned step into staging, a quota per step, so a step's L1 reads overlap the steps in between. Nominal
//   x_step_rounds * core_row_tiles_max * k_block_tiles pages pushed per step; an empty m-group moves no data.
// x_step_rounds: one x step carries that many consecutive compute K rounds (x K block = x_step_rounds *
//   k_block_tiles), laid out round-major ([round][row][k]) so compute pops them as separate rounds. Fewer, larger x
//   steps keep the x chain's per-step cost while the w rounds (and their CBs) stay at k_block_tiles.
// w_reads_on_reader: this RISC also runs the w line sender's DRAM prefetch (moe_down_proj_w_prefetch.h) on NoC1 into
//   cb_w_stage, a quota per step once the writer RISC has freed the stage, and pushes it one step before the owned
//   w step; the writer RISC multicasts it on NoC0.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "moe_down_proj_plan.h"
#include "moe_down_proj_store.h"
#include "moe_down_proj_w_prefetch.h"

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
    constexpr uint32_t w_reader_share = get_compile_time_arg_val(20);
    constexpr bool w_reads_on_reader = w_reader_share > 0;
    constexpr uint32_t w_share_den = get_compile_time_arg_val(21);
    constexpr uint32_t cb_w_stage = get_compile_time_arg_val(22);
    constexpr uint32_t cb_w_owner = get_compile_time_arg_val(23);
    constexpr uint32_t x_step_rounds = get_compile_time_arg_val(24);
    constexpr uint32_t cb_tail = get_compile_time_arg_val(25);
    static_assert(num_k_rounds % x_step_rounds == 0, "x steps must not straddle a wave");
    static_assert(x_step_rounds == 1 || !w_reads_on_reader, "the reader's w prefetch steps per w round");
    constexpr uint32_t x_steps_per_wave = num_k_rounds / x_step_rounds;
    constexpr auto x_args = TensorAccessorArgs<26>();
    constexpr auto counts_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto out_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto w_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto xa_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("xa_ct_offset"), get_named_compile_time_arg_val("xa_rt_offset")>();
    constexpr auto xb_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("xb_ct_offset"), get_named_compile_time_arg_val("xb_rt_offset")>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_arg_val<uint32_t>(1);
    const uint32_t group = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t line = get_arg_val<uint32_t>(4);
    const uint32_t w_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t x_round_tiles = core_row_tiles_max * k_block_tiles;
    constexpr uint32_t x_step_tiles = x_step_rounds * x_round_tiles;
    constexpr uint32_t x_step_k_tiles = x_step_rounds * k_block_tiles;

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
        num_steps += x_steps_per_wave;
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
    constexpr uint32_t issue_quota = (x_step_tiles + prefetch_issue_steps - 1) / prefetch_issue_steps;
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
        const uint32_t wave = step / x_steps_per_wave;
        pf_first_page = plan.group_first(wave, group) * tensor_k_tiles + (step % x_steps_per_wave) * x_step_k_tiles;
        pf_total = plan.group_rows(wave, group) * x_step_k_tiles;
    };
    auto issue_prefetch = [&](uint32_t max_tiles) {
        for (; pf_next < pf_total && max_tiles > 0; ++pf_next, --max_tiles) {
            const uint32_t row = pf_next / x_step_k_tiles;
            const uint32_t kk = pf_next % x_step_k_tiles;
            const uint32_t round = kk / k_block_tiles;
            const uint32_t stage_tile = round * x_round_tiles + row * k_block_tiles + kk % k_block_tiles;
#ifndef ABL_NO_X
            noc_async_read_page(
                pf_first_page + row * tensor_k_tiles + kk, x_accessor, stage_l1 + stage_tile * tile_bytes);
#else
            (void)stage_tile;
#endif
        }
    };

    // Wave p's store starts after wave p + 1's first x_store_round x rounds (the x slots compute can fill before it
    // has untilized wave p), so compute starts wave p + 1 while the store drains, store_blocks_per_round per round.
    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_bytes);
    constexpr uint32_t x_store_round = x_depth < x_steps_per_wave ? x_depth : x_steps_per_wave;
    auto store_wave = [&](uint32_t wave, uint32_t first = 0, uint32_t max_blocks = UINT32_MAX) {
        if constexpr (store_on_reader) {
            return moe_store_wave<cb_output_rows, core_n_tiles>(
                plan, out_accessor, wave, group, line, first, max_blocks);
        }
        return first;
    };

    // x and w staged reads carry their own transaction ids, so neither channel's barrier waits on the other's reads.
    constexpr uint32_t X_TRID = 1;
    constexpr uint32_t W_TRID = 2;
    dataflow_kernel_lib::set_read_trid(noc, X_TRID);
    constexpr uint32_t w_round_tiles = MOE_NUM_EXPERTS * k_block_tiles * core_n_tiles;
    constexpr uint32_t w_issue_quota = (w_round_tiles + prefetch_issue_steps - 1) / prefetch_issue_steps;
    const auto w_accessor = TensorAccessor(w_args, w_addr, tile_bytes);
    MoeWPrefetch<
        k_block_tiles,
        core_n_tiles,
        tensor_k_tiles,
        num_k_rounds,
        tile_bytes,
        w_share_den,
        decltype(w_accessor)>
        w_prefetch(plan, w_accessor, line * core_n_tiles, 0, w_reader_share);
    MoeWOwner w_owns{MOE_W_OWNER_NONE, 1};
    bool w_reserved = false;
    // Before x step `step`: advance the w prefetch, and complete it (barrier + push) one step before its owned step.
    auto w_stage_step = [&](uint32_t step) {
        if (w_prefetch.step == w_prefetch.NONE) {
            return;
        }
        const bool due = w_prefetch.step <= step + 1;
        if (!w_reserved && (due || cb_pages_reservable_at_back(cb_w_stage, w_round_tiles))) {
            cb_reserve_back(cb_w_stage, w_round_tiles);
            w_reserved = true;
        }
        if (!w_reserved) {
            return;
        }
        dataflow_kernel_lib::set_read_trid(noc, W_TRID);
        w_prefetch.issue(get_write_ptr(cb_w_stage), due ? w_prefetch.total : w_issue_quota);
        dataflow_kernel_lib::set_read_trid(noc, X_TRID);
        if (due) {
            MaybeDeviceZoneScope("w_stage_complete");
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, W_TRID);
            cb_push_back(cb_w_stage, w_round_tiles);
            w_reserved = false;
            w_prefetch.start(w_prefetch.step + 1, num_steps, w_owns);
        }
    };
    if constexpr (w_reads_on_reader) {
        w_owns = moe_receive_w_owner<cb_w_owner>();
        w_prefetch.start(0, num_steps, w_owns);
    }

    // L layout: the xb fixed sender forwards the last received step (pending) before acking the next one.
    const bool forwarder = xb_mcast.can_send();
    uint32_t pending_slot = 0;
    uint32_t pending_bytes = 0;
    auto forward_pending = [&]() {
        if (pending_bytes == 0) {
            return;
        }
        // The previous forward read the slot this step's ack frees.
        noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = 0});
        xb_sender->template send<dataflow_kernel_lib::SourceL1Guard::CallerManaged>(
            pending_slot, pending_slot, pending_bytes);
        pending_bytes = 0;
    };

    start_prefetch(0);
    uint32_t stored = 0;
    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t round = step % x_steps_per_wave;
        if (round == 0) {
            stored = 0;
        }
        if (step >= x_steps_per_wave && round >= x_store_round) {
            stored = store_wave(step / x_steps_per_wave - 1, stored, store_blocks_per_round);
        }
        if constexpr (w_reads_on_reader) {
            w_stage_step(step);
        }
        const uint32_t group_rows = plan.group_rows(step / x_steps_per_wave, group);
        const bool owned = on_xa && xa_mcast.should_send(step);
        issue_prefetch(owned ? pf_total : issue_quota);
        {
            MaybeDeviceZoneScope("x_reserve");
            cb_reserve_back(cb_x_round, x_step_tiles);
        }
        if (forwarder) {
            MaybeDeviceZoneScope("x_forward");
            forward_pending();
        }
        if (group_rows > 0) {
            const uint32_t slot = get_write_ptr(cb_x_round);
            // Every round but the last spans its nominal slot; the last ends at the group's rows.
#ifdef ABL_NO_X
            const uint32_t bytes = tile_bytes;
#else
            const uint32_t bytes = ((x_step_rounds - 1) * x_round_tiles + group_rows * k_block_tiles) * tile_bytes;
#endif
            if (owned) {
                MaybeDeviceZoneScope("x_send");
                {
                    MaybeDeviceZoneScope("x_send_barrier");
                    dataflow_kernel_lib::async_read_barrier_with_trid(noc, X_TRID);
                }
                xa_sender->send(stage_l1, slot, bytes);
            } else if (on_xa) {
                {
                    MaybeDeviceZoneScope("x_receive");
                    xa_receiver->receive(step);
                }
                if (forwarder) {
                    pending_slot = slot;
#ifdef ABL_NO_STRIP
                    pending_bytes = tile_bytes;
#else
                    pending_bytes = bytes;
#endif
                }
            } else {
                MaybeDeviceZoneScope("x_receive");
                xb_receiver->receive(step);
            }
        }
        if (owned) {
            start_prefetch(step + 1);
        }
        cb_push_back(cb_x_round, x_step_tiles);
        if (step >= x_steps_per_wave && round == x_steps_per_wave - 1) {
            store_wave(step / x_steps_per_wave - 1, stored);
        }
    }
    if (forwarder) {
        forward_pending();
        noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = 0});
    }
    if (num_steps > 0) {
        const uint32_t last_wave = num_steps / x_steps_per_wave - 1;
        if constexpr (store_on_reader) {
            moe_store_tail_owner<cb_output_rows, cb_tail, core_n_tiles>(plan, out_accessor, last_wave, group, line);
        } else {
            moe_store_tail_helper<cb_tail, core_n_tiles>(plan, out_accessor, last_wave, group, line);
        }
    }
    noc_async_write_barrier();
}
