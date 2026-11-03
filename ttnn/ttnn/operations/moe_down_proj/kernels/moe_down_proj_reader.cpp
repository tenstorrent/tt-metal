// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj reader (both multicast channels on this RISC).
// D0: counts page -> cb_counts_compute and cb_counts_writer (parsed from the landing copy for the plan).
// Per wave, per K round, in the global channel-round order shared by every core:
//   deliver_x_round: the round's rotating sender of the m-group reads its rows' K block of x (one barrier) and
//     multicasts it along the m-group (xa); in the L layout it also copies the same slot to the group's strip
//     core (xb). Nominal core_row_tiles_max * k_block_tiles pages pushed; an empty m-group moves no data.
//   deliver_w_round: per in-wave expert (expert order), the n-line's rotating sender reads the expert's
//     k_block_tiles x core_n_tiles chunk ([k][n] order) into the expert's slot and multicasts it along the line
//     (wa: rect lines, wb: the L strip line). All NUM_EXPERTS slots are pushed, empty ones without data.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "moe_down_proj_plan.h"

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_round = get_compile_time_arg_val(1);
    constexpr uint32_t cb_counts_compute = get_compile_time_arg_val(2);
    constexpr uint32_t cb_counts_writer = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t num_waves = get_compile_time_arg_val(5);
    constexpr uint32_t num_m_groups = get_compile_time_arg_val(6);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t num_k_rounds = get_compile_time_arg_val(8);
    constexpr uint32_t core_row_tiles_max = get_compile_time_arg_val(9);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t w_row_pages = get_compile_time_arg_val(12);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(13);
    constexpr uint32_t counts_page_bytes = get_compile_time_arg_val(14);
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto xa_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("xa_ct_offset"), get_named_compile_time_arg_val("xa_rt_offset")>();
    constexpr auto xb_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("xb_ct_offset"), get_named_compile_time_arg_val("xb_rt_offset")>();
    constexpr auto wa_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wa_ct_offset"), get_named_compile_time_arg_val("wa_rt_offset")>();
    constexpr auto wb_mcast = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("wb_ct_offset"), get_named_compile_time_arg_val("wb_rt_offset")>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t counts_addr = get_arg_val<uint32_t>(2);
    const uint32_t group = get_arg_val<uint32_t>(3);
    const uint32_t line = get_arg_val<uint32_t>(4);

    constexpr uint32_t x_round_tiles = core_row_tiles_max * k_block_tiles;
    constexpr uint32_t w_chunk_tiles = k_block_tiles * core_n_tiles;
    constexpr uint32_t w_round_tiles = MOE_NUM_EXPERTS * w_chunk_tiles;

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto w_accessor = TensorAccessor(w_args, w_addr, tile_bytes);
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

    Noc noc;
    auto xa_sender = xa_mcast.optional_sender(noc);
    auto xa_receiver = xa_mcast.optional_receiver(noc);
    auto xb_sender = xb_mcast.optional_sender(noc);
    auto xb_receiver = xb_mcast.optional_receiver(noc);
    auto wa_sender = wa_mcast.optional_sender(noc);
    auto wa_receiver = wa_mcast.optional_receiver(noc);
    auto wb_sender = wb_mcast.optional_sender(noc);
    auto wb_receiver = wb_mcast.optional_receiver(noc);
    // Rect cores own an xa seat (and send xb in the L layout); L strip cores only receive xb.
    const bool on_xa = xa_mcast.can_send() || xa_mcast.can_receive();
    // Every core sits in exactly one n-line: a wa line (rect) or the wb strip line.
    const bool on_wa = wa_mcast.can_send() || wa_mcast.can_receive();
    const uint32_t n_first = line * core_n_tiles;

    uint32_t x_round = 0;
    uint32_t w_chunk = 0;
    for (uint32_t wave = 0; wave < num_waves; ++wave) {
        if (plan.wave_empty(wave)) {
            continue;
        }
        const uint32_t group_first = plan.group_first(wave, group);
        const uint32_t group_rows = plan.group_rows(wave, group);
        bool in_wave[MOE_NUM_EXPERTS];
        for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
            in_wave[e] = plan.expert_in_wave(wave, e);
        }

        for (uint32_t round = 0; round < num_k_rounds; ++round, ++x_round) {
            const uint32_t k_first = round * k_block_tiles;

            // deliver_x_round
            cb_reserve_back(cb_x_round, x_round_tiles);
            if (group_rows > 0) {
                const uint32_t slot = get_write_ptr(cb_x_round);
                const uint32_t bytes = group_rows * k_block_tiles * tile_bytes;
                if (on_xa && xa_mcast.should_send(x_round)) {
                    for (uint32_t r = 0; r < group_rows; ++r) {
                        const uint32_t page = (group_first + r) * tensor_k_tiles + k_first;
                        const uint32_t row_l1 = slot + r * k_block_tiles * tile_bytes;
                        for (uint32_t kk = 0; kk < k_block_tiles; ++kk) {
                            noc_async_read_page(page + kk, x_accessor, row_l1 + kk * tile_bytes);
                        }
                    }
                    noc_async_read_barrier();
                    xa_sender->send(slot, slot, bytes);
                    if (xb_mcast.can_send()) {
                        xb_sender->send(slot, slot, bytes);
                    }
                } else if (on_xa) {
                    xa_receiver->receive(x_round);
                } else {
                    xb_receiver->receive(x_round);
                }
            }
            cb_push_back(cb_x_round, x_round_tiles);

            // deliver_w_round
            cb_reserve_back(cb_w_round, w_round_tiles);
            const uint32_t w_base = get_write_ptr(cb_w_round);
            for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
                if (in_wave[e]) {
                    const uint32_t chunk_l1 = w_base + e * w_chunk_tiles * tile_bytes;
                    const bool send = on_wa ? wa_mcast.should_send(w_chunk) : wb_mcast.should_send(w_chunk);
                    if (send) {
                        for (uint32_t n = 0; n < core_n_tiles; ++n) {
                            const uint32_t page = (n_first + n) * w_row_pages + e * tensor_k_tiles + k_first;
                            for (uint32_t kk = 0; kk < k_block_tiles; ++kk) {
                                noc_async_read_page(
                                    page + kk, w_accessor, chunk_l1 + (kk * core_n_tiles + n) * tile_bytes);
                            }
                        }
                        noc_async_read_barrier();
                        if (on_wa) {
                            wa_sender->send(chunk_l1, chunk_l1, w_chunk_tiles * tile_bytes);
                        } else {
                            wb_sender->send(chunk_l1, chunk_l1, w_chunk_tiles * tile_bytes);
                        }
                    } else if (on_wa) {
                        wa_receiver->receive(w_chunk);
                    } else {
                        wb_receiver->receive(w_chunk);
                    }
                    ++w_chunk;
                }
                cb_push_back(cb_w_round, w_chunk_tiles);
            }
        }
    }
}
