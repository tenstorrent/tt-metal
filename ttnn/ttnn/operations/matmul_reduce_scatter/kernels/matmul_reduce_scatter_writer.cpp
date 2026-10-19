// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — compute-core BRISC: weight (W, in1) operand.
//
// Per compute unit (one row-wave of a scatter block, in compute order) and per K-block: the n-line injector (W_SENDS)
// reads its columns' K-block from DRAM and multicasts it along its n-line (mcast_pipe SenderPipe); the other cores of
// the line receive it. Each unit carries its W column origin and a `fresh` flag from the host; a non-fresh unit only
// replays CB credits (capacity = one K pass exactly, so the ring wraps onto the same pages): W resident for the call
// (scatter_dim=-2, R1: every unit after the first), or W held across one block's waves (scatter_dim=-1 with waves:
// waves 1.. of each block).

#include <cstdint>
#include <optional>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "matmul_reduce_scatter_injector.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_weight_operand = get_compile_time_arg_val(0);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);  // compute units (G blocks x waves)
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t w_sends = get_compile_time_arg_val(6);          // 0 receive, 1 fixed injector, 2 rotating sender
    constexpr bool read_ahead = get_compile_time_arg_val(7) != 0;      // injector DRAM read-ahead (injector header)
    constexpr bool rows_same_bank = get_compile_time_arg_val(8) != 0;  // Nt % DRAM banks == 0
    constexpr auto w_args = TensorAccessorArgs<9>();
    constexpr auto mc_w =
        McastArgs<get_named_compile_time_arg_val("w_ct_offset"), get_named_compile_time_arg_val("w_rt_offset")>();

    constexpr uint32_t kblock_pages = k_block_tiles * core_n_tiles;
    constexpr uint32_t kblock_bytes = kblock_pages * w_tile_bytes;

    size_t arg = 0;
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_row_stride = get_arg_val<uint32_t>(arg++);  // Nt (W pages per tile-row)
    const uint32_t w_col0 = get_arg_val<uint32_t>(arg++);        // n-line's first column within a block
    const uint32_t w_valid_cols = get_arg_val<uint32_t>(arg++);  // <= core_n_tiles (ragged last n-line)
    const uint32_t order_idx = arg;                              // num_blocks x [unit's W column origin, W fresh]
    arg += 2 * num_blocks;

    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);

    Noc noc;
    // K-block steps: step s = (unit s / num_k_blocks, K-block s % num_k_blocks)
    constexpr uint32_t steps = num_blocks * num_k_blocks;
    auto fresh = [&](uint32_t s) { return get_arg_val<uint32_t>(order_idx + 2 * (s / num_k_blocks) + 1) != 0; };
    auto reserve = [&](uint32_t pages) { cb_reserve_back(cb_weight_operand, pages); };
    auto no_poll = []() {};
    // CB layout per K-block: [k_block_tiles][core_n_tiles]; padding columns stay unread. Read column by column:
    // a column's K-block rows are Nt pages apart, i.e. one bank when Nt is a multiple of the bank count.
    auto issue_w = [&](uint32_t s, uint32_t dst) {
        const uint32_t b = s / num_k_blocks, kb = s - b * num_k_blocks;
        const uint32_t page0 = kb * k_block_tiles * w_row_stride + get_arg_val<uint32_t>(order_idx + 2 * b) + w_col0;
        for (uint32_t c = 0; c < w_valid_cols; ++c) {
            mmrs::read_pages_strided(
                w_acc,
                page0 + c,
                w_row_stride,
                k_block_tiles,
                dst + c * w_tile_bytes,
                core_n_tiles * w_tile_bytes,
                w_tile_bytes,
                rows_same_bank);
        }
    };
    if constexpr (w_sends == 2) {
        // Rotating senders: every core of the n-line reads + multicasts every span-th fresh K-block (its own DRAM
        // reads overlap the others' multicasts). The round is offset by the line index so the senders of one round
        // sit on a diagonal of the grid, not in one row / column.
        using SendPipe = decltype(mc_w.sender(noc));
        using RecvPipe = decltype(mc_w.receiver(noc));
        std::optional<SendPipe> send_pipe;
        std::optional<RecvPipe> recv_pipe;
        if (mc_w.can_send()) {
            send_pipe.emplace(mc_w.sender(noc));
        }
        if (mc_w.can_receive()) {
            recv_pipe.emplace(mc_w.receiver(noc));
        }
        const uint32_t line_off = get_arg_val<uint32_t>(order_idx + 2 * num_blocks);
        // Resident W (fresh only in the first unit): every K-block owns its own CB slot (capacity = one K pass) and
        // the CB is empty at start, so each sender issues its resident K-block reads up front, in parallel with the
        // other senders' reads; the multicasts then go out in round order.
        const bool w_resident = num_blocks > 1 && !fresh(num_k_blocks);
        if (w_resident) {
            const uint32_t base = get_write_ptr(cb_weight_operand);
            for (uint32_t s = 0; s < num_k_blocks; ++s) {
                if (mc_w.should_send(line_off + s)) {
                    issue_w(s, base + s * kblock_bytes);
                }
            }
        }
        uint32_t round = line_off;
        for (uint32_t s = 0; s < steps; ++s) {
            {
                MaybeDeviceZoneScope("recv_reserve");
                reserve(kblock_pages);
            }
            if (fresh(s)) {
                const uint32_t dst = get_write_ptr(cb_weight_operand);
                if (mc_w.should_send(round)) {
                    {
                        MaybeDeviceZoneScope("inj_read");
                        if (!w_resident) {
                            issue_w(s, dst);
                        }
                        noc_async_read_barrier();
                    }
                    MaybeDeviceZoneScope("inj_mcast");
                    send_pipe->send(dst, dst, kblock_bytes);
                } else {
                    MaybeDeviceZoneScope("recv_mcast");
                    recv_pipe->receive(round);
                }
                ++round;
            }
            cb_push_back(cb_weight_operand, kblock_pages);
        }
    } else if constexpr (w_sends) {
        auto pipe = mc_w.sender(noc);
        mmrs::inject_operand<read_ahead>(
            cb_weight_operand,
            steps,
            kblock_pages,
            kblock_bytes,
            fresh,
            issue_w,
            reserve,
            [&](uint32_t dst) { pipe.send(dst, dst, kblock_bytes); },
            no_poll);
    } else {
        auto pipe = mc_w.receiver(noc);
        mmrs::receive_operand(
            cb_weight_operand, steps, kblock_pages, fresh, reserve, [&]() { pipe.receive(); }, no_poll);
    }
}
