// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr std::uint32_t output_cb_id = get_compile_time_arg_val(0);
constexpr auto dst_args = TensorAccessorArgs<1>();
constexpr std::uint32_t state_ct_base = dst_args.next_compile_time_args_offset();
constexpr bool has_state = get_compile_time_arg_val(state_ct_base) != 0;
constexpr std::uint32_t state_cb_id = get_compile_time_arg_val(state_ct_base + 1);
constexpr std::uint32_t state_page_bytes = 128;  // one uint32 row of 32 words per core

// The state accessor's compile-time args exist only when a state tensor was bound; `Cta` keeps the accessor
// type dependent so the disabled instantiation never names them.
template <bool Enabled, std::uint32_t Cta>
struct StatePage {
    std::uint32_t epoch_lo = 0;
    std::uint32_t epoch_hi = 0;

    // Hand this core's epoch to compute before any tile is produced.
    void read(Noc& noc) {
        if constexpr (Enabled) {
            constexpr auto state_args = TensorAccessorArgs<Cta>();
            const auto state_addrg = TensorAccessor(state_args, get_arg_val<std::uint32_t>(3));
            CircularBuffer cb_state(state_cb_id);
            cb_state.reserve_back(1);
            noc.async_read(
                state_addrg,
                cb_state,
                state_page_bytes,
                {.page_id = get_arg_val<std::uint32_t>(4)},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            CoreLocalMem<volatile std::uint32_t> words(cb_state.get_write_ptr());
            epoch_lo = words[0];
            epoch_hi = words[1];
            cb_state.push_back(1);
        }
    }

    void write_back(Noc& noc) {
        if constexpr (Enabled) {
            constexpr auto state_args = TensorAccessorArgs<Cta>();
            const auto state_addrg = TensorAccessor(state_args, get_arg_val<std::uint32_t>(3));
            CircularBuffer cb_state(state_cb_id);
            cb_state.reserve_back(1);
            CoreLocalMem<volatile std::uint32_t> words(cb_state.get_write_ptr());
            for (std::uint32_t w = 0; w < state_page_bytes / sizeof(std::uint32_t); ++w) {
                words[w] = 0;
            }
            words[0] = epoch_lo + 1;
            words[1] = epoch_hi + (epoch_lo == 0xFFFFFFFFu ? 1 : 0);
            noc.async_write(
                CoreLocalMem<std::uint32_t>(cb_state.get_write_ptr()),
                state_addrg,
                state_page_bytes,
                {},
                {.page_id = get_arg_val<std::uint32_t>(4)});
            noc.async_write_barrier();
        }
    }
};

void kernel_main() {
    std::uint32_t dst_addr = get_arg_val<std::uint32_t>(0);
    std::uint32_t start_id = get_arg_val<std::uint32_t>(1);
    std::uint32_t num_tiles = get_arg_val<std::uint32_t>(2);
    std::uint32_t end_id = start_id + num_tiles;

    const auto output_addrg = TensorAccessor(dst_args, dst_addr);

    const std::uint32_t page_bytes = get_local_cb_interface(output_cb_id).fifo_page_size;

    Noc noc;
    CircularBuffer cb_output(output_cb_id);
    StatePage<has_state, has_state ? state_ct_base + 2 : 1> state;
    state.read(noc);

    for (std::uint32_t i = start_id; i < end_id; ++i) {
        cb_output.wait_front(1);
        std::uint32_t output_cb_read_ptr = cb_output.get_read_ptr();
        noc.async_write(CoreLocalMem<std::uint32_t>(output_cb_read_ptr), output_addrg, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb_output.pop_front(1);
    }
    noc.async_write_barrier();
    state.write_back(noc);
}
