// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// x broadcast + y store. Every grid core walks the num_k_blocks multicast rounds in order: the round's
// injector (rotating sender) multicasts its prefetched slot in place, every other core waits on the
// monotone Counter. No ready handshake: each round lands in its own write-once slot of cb_x_resident.
// Then the core writes its (block_m_tiles x core_n_tiles) y block to L1-interleaved y.

#include <cstdint>
#include <optional>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto x_mcast = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_start = get_arg_val<uint32_t>(2);
    const uint32_t core_n_tiles = get_arg_val<uint32_t>(3);

    const uint32_t x_page_bytes = get_tile_size(cb_x_resident);
    const uint32_t y_page_bytes = get_tile_size(cb_y_tiles);
    const auto x = TensorAccessor(x_args, x_addr, x_page_bytes);
    const auto y = TensorAccessor(y_args, y_addr, y_page_bytes);

    constexpr uint32_t round_tiles = block_m_tiles * k_block_tiles;
    const uint32_t round_bytes = round_tiles * x_page_bytes;

    Noc noc;
    CircularBuffer cb_x(cb_x_resident);
    cb_x.reserve_back(round_tiles * num_k_blocks);
    const uint32_t x_base = cb_x.get_write_ptr();

    using SendPipe = decltype(x_mcast.sender(noc));
    using ReceivePipe = decltype(x_mcast.receiver(noc));
    std::optional<SendPipe> send_pipe;
    std::optional<ReceivePipe> recv_pipe;
    const bool is_injector = x_mcast.can_send();
    if (is_injector) {
        send_pipe.emplace(x_mcast.sender(noc));
        for (uint32_t r = 0; r < num_k_blocks; ++r) {
            if (!x_mcast.should_send(r)) {
                continue;
            }
            uint32_t offset = r * round_bytes;
            for (uint32_t m = 0; m < block_m_tiles; ++m) {
                const uint32_t row_page = m * tensor_k_tiles + r * k_block_tiles;
                for (uint32_t kk = 0; kk < k_block_tiles; ++kk) {
                    noc.async_read(x, cb_x, x_page_bytes, {.page_id = row_page + kk}, {.offset_bytes = offset});
                    offset += x_page_bytes;
                }
            }
        }
    }
    if (x_mcast.can_receive()) {
        recv_pipe.emplace(x_mcast.receiver(noc));
    }

    bool prefetch_landed = false;
    for (uint32_t r = 0; r < num_k_blocks; ++r) {
        if (x_mcast.should_send(r)) {
            if (!prefetch_landed) {
                noc.async_read_barrier();
                prefetch_landed = true;
            }
            const uint32_t slot = x_base + r * round_bytes;
            send_pipe->send(slot, slot, round_bytes);
        } else {
            recv_pipe->receive(r);
        }
        cb_x.push_back(round_tiles);
    }

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
            noc.async_write(cb_y, y, y_page_bytes, {.offset_bytes = offset}, {.page_id = row_page + n});
            offset += y_page_bytes;
        }
    }
    noc.async_write_barrier();
    cb_y.pop_front(y_block_tiles);
}
