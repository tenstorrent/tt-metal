// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// x broadcast + y store. x is cut into num_x_rounds multicast rounds of x_round_k_blocks compute K-blocks.
// Each round's injector prefetches its slot and multicasts it in place; every other core receives it. Each
// round lands in its own write-once slot of cb_x_resident, laid out as consecutive
// (block_m_tiles x k_block_tiles) K-blocks, and is released to compute per K-block.
//   per_round: round r travels on its own fixed-sender channel x_mcast_r, fired once per launch.
//   rotating:  all rounds travel on x_mcast_0 with rotating senders.
// Chained rounds go out in order (the round-r injector has received round r - 1 first); concurrent rounds go
// out at once. Then the core writes its (block_m_tiles x core_n_tiles) y block to L1-interleaved y.
// With w_noc1_k_blocks > 0 this kernel also reads the last w_noc1_k_blocks W K-blocks on NoC1 into their
// cb_w_stream slots (the reader owns the CB and pushes them) and raises the reader's handoff semaphore once
// they landed; the x prefetch then waits on its own trid instead of every outstanding read. The reads go out
// with the x prefetch, or after the core's x phase when w_noc1_after_x (keeping NoC1 to the broadcast).

#include <cstdint>
#include <optional>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/ttnn/operations/wq_b_linear/kernels/wq_b_linear_w_read.hpp"

using namespace dataflow_kernel_lib;

template <uint32_t I>
constexpr auto x_channel();

#define X_MCAST_CHANNEL(I)                                                 \
    template <>                                                            \
    constexpr auto x_channel<I>() {                                        \
        return McastArgs<                                                  \
            get_named_compile_time_arg_val("x_mcast_" #I "_ct_offset"),    \
            get_named_compile_time_arg_val("x_mcast_" #I "_rt_offset")>(); \
    }
X_MCAST_CHANNEL(0)
X_MCAST_CHANNEL(1)
X_MCAST_CHANNEL(2)
X_MCAST_CHANNEL(3)
X_MCAST_CHANNEL(4)
X_MCAST_CHANNEL(5)
X_MCAST_CHANNEL(6)
X_MCAST_CHANNEL(7)
constexpr uint32_t num_declared_x_channels = 8;

template <uint32_t I, uint32_t N, typename F>
FORCE_INLINE void for_each_x_channel(F& f) {
    if constexpr (I < N) {
        f(x_channel<I>(), std::integral_constant<uint32_t, I>{});
        for_each_x_channel<I + 1, N>(f);
    }
}

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t x_round_k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t num_x_rounds = get_compile_time_arg_val(6);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(7);
    constexpr bool x_channel_per_round = get_compile_time_arg_val(8) != 0;
    constexpr bool x_rounds_concurrent = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t max_x_channels = get_compile_time_arg_val(10);
    constexpr uint32_t cb_w_stream = get_compile_time_arg_val(11);
    constexpr uint32_t w_noc1_k_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t w_handoff_sem_id = get_compile_time_arg_val(13);
    constexpr bool w_noc1_after_x = get_compile_time_arg_val(14) != 0;
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t x_prefetch_trid = 1;
    constexpr uint32_t w_trid = 2;
    static_assert(max_x_channels == num_declared_x_channels);
    static_assert(!x_channel_per_round || num_x_rounds <= max_x_channels);

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_start = get_arg_val<uint32_t>(2);
    const uint32_t core_n_tiles = get_arg_val<uint32_t>(3);
    const uint32_t n_stride = get_arg_val<uint32_t>(4);
    const uint32_t w_addr = get_arg_val<uint32_t>(5);

    const uint32_t x_page_bytes = get_tile_size(cb_x_resident);
    const uint32_t y_page_bytes = get_tile_size(cb_y_tiles);
    const auto x = TensorAccessor(x_args, x_addr, x_page_bytes);
    const auto y = TensorAccessor(y_args, y_addr, y_page_bytes);

    constexpr uint32_t k_block_x_tiles = block_m_tiles * k_block_tiles;
    constexpr uint32_t round_tiles = k_block_x_tiles * x_round_k_blocks;
    const uint32_t round_bytes = round_tiles * x_page_bytes;

    Noc noc;
    CircularBuffer cb_x(cb_x_resident);
    cb_x.reserve_back(round_tiles * num_x_rounds);
    const uint32_t x_base = cb_x.get_write_ptr();

    const bool reads_w = w_noc1_k_blocks > 0 && core_n_tiles > 0;
    bool x_phase_done = false;
    auto issue_w_reads = [&]() {
        if (!reads_w || x_phase_done != w_noc1_after_x) {
            return;
        }
        constexpr uint32_t num_k_blocks = tensor_k_tiles / k_block_tiles;
        const uint32_t w_page_bytes = get_tile_size(cb_w_stream);
        const auto w = TensorAccessor(w_args, w_addr, w_page_bytes);
        CircularBuffer cb_w(cb_w_stream);
        const uint32_t block_bytes = k_block_tiles * core_n_tiles * w_page_bytes;
        dataflow_kernel_lib::set_read_trid(noc, w_trid);
        for (uint32_t b = num_k_blocks - w_noc1_k_blocks; b < num_k_blocks; ++b) {
            read_w_block(
                noc,
                w,
                cb_w,
                w_page_bytes,
                b,
                b * block_bytes,
                k_block_tiles,
                tensor_n_tiles,
                n_start,
                core_n_tiles,
                n_stride);
        }
        dataflow_kernel_lib::set_read_trid(noc, 0);
    };
    auto hand_off_w = [&]() {
        if (reads_w) {
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, w_trid);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(w_handoff_sem_id)), 1);
        }
    };

    auto prefetch_round = [&](uint32_t r) {
        if constexpr (w_noc1_k_blocks > 0) {
            dataflow_kernel_lib::set_read_trid(noc, x_prefetch_trid);
        }
        uint32_t offset = r * round_bytes;
        for (uint32_t b = r * x_round_k_blocks; b < (r + 1) * x_round_k_blocks; ++b) {
            for (uint32_t m = 0; m < block_m_tiles; ++m) {
                const uint32_t row_page = m * tensor_k_tiles + b * k_block_tiles;
                for (uint32_t kk = 0; kk < k_block_tiles; ++kk) {
                    noc.async_read(x, cb_x, x_page_bytes, {.page_id = row_page + kk}, {.offset_bytes = offset});
                    offset += x_page_bytes;
                }
            }
        }
        if constexpr (w_noc1_k_blocks > 0) {
            dataflow_kernel_lib::set_read_trid(noc, 0);
        }
    };
    bool prefetch_landed = false;
    auto send_round = [&](auto& send_pipe, uint32_t r) {
        if (!prefetch_landed) {
            if constexpr (w_noc1_k_blocks > 0) {
                dataflow_kernel_lib::async_read_barrier_with_trid(noc, x_prefetch_trid);
            } else {
                noc.async_read_barrier();
            }
            prefetch_landed = true;
        }
        const uint32_t slot = x_base + r * round_bytes;
        send_pipe.send(slot, slot, round_bytes);
    };
    auto release_round = [&]() {
        for (uint32_t b = 0; b < x_round_k_blocks; ++b) {
            cb_x.push_back(k_block_x_tiles);
        }
    };

    if constexpr (x_channel_per_round) {
        auto prefetch = [&](auto ch, auto r) {
            if (ch.can_send()) {
                prefetch_round(r);
            }
        };
        for_each_x_channel<0, num_x_rounds>(prefetch);
        issue_w_reads();
        if constexpr (x_rounds_concurrent) {
            auto send = [&](auto ch, auto r) {
                if (ch.can_send()) {
                    auto pipe = ch.sender(noc);
                    send_round(pipe, r);
                }
            };
            for_each_x_channel<0, num_x_rounds>(send);
            auto receive = [&](auto ch, auto) {
                if (ch.can_receive()) {
                    ch.receiver(noc).receive(0);
                }
                release_round();
            };
            for_each_x_channel<0, num_x_rounds>(receive);
        } else {
            auto send_or_receive = [&](auto ch, auto r) {
                if (ch.can_send()) {
                    auto pipe = ch.sender(noc);
                    send_round(pipe, r);
                } else if (ch.can_receive()) {
                    ch.receiver(noc).receive(0);
                }
                release_round();
            };
            for_each_x_channel<0, num_x_rounds>(send_or_receive);
        }
    } else {
        constexpr auto x_mcast = x_channel<0>();
        using SendPipe = decltype(x_mcast.sender(noc));
        using ReceivePipe = decltype(x_mcast.receiver(noc));
        std::optional<SendPipe> send_pipe;
        std::optional<ReceivePipe> recv_pipe;
        if (x_mcast.can_send()) {
            send_pipe.emplace(x_mcast.sender(noc));
            for (uint32_t r = 0; r < num_x_rounds; ++r) {
                if (x_mcast.should_send(r)) {
                    prefetch_round(r);
                }
            }
        }
        issue_w_reads();
        if (x_mcast.can_receive()) {
            recv_pipe.emplace(x_mcast.receiver(noc));
        }
        if constexpr (x_rounds_concurrent) {
            for (uint32_t r = 0; r < num_x_rounds; ++r) {
                if (x_mcast.should_send(r)) {
                    send_round(*send_pipe, r);
                }
            }
            if (recv_pipe) {
                recv_pipe->receive(num_x_rounds - 1);
            }
            for (uint32_t r = 0; r < num_x_rounds; ++r) {
                release_round();
            }
        } else {
            for (uint32_t r = 0; r < num_x_rounds; ++r) {
                if (x_mcast.should_send(r)) {
                    send_round(*send_pipe, r);
                } else {
                    recv_pipe->receive(r);
                }
                release_round();
            }
        }
    }

    x_phase_done = true;
    issue_w_reads();
    hand_off_w();
    if (core_n_tiles == 0) {
        return;
    }
    CircularBuffer cb_y(cb_y_tiles);
    const uint32_t y_block_tiles = block_m_tiles * core_n_tiles;
    cb_y.wait_front(y_block_tiles);
    uint32_t offset = 0;
    for (uint32_t m = 0; m < block_m_tiles; ++m) {
        const uint32_t row_page = m * tensor_n_tiles + n_start;
        for (uint32_t n = 0; n < core_n_tiles; ++n) {
            noc.async_write(cb_y, y, y_page_bytes, {.offset_bytes = offset}, {.page_id = row_page + n * n_stride});
            offset += y_page_bytes;
        }
    }
    noc.async_write_barrier();
    cb_y.pop_front(y_block_tiles);
}
