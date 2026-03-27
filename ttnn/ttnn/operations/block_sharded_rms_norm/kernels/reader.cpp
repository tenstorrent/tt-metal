// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

FORCE_INLINE void copy_tile_local(uint32_t src_addr, uint32_t dst_addr, uint32_t tile_nbytes) {
    auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t i = 0; i < tile_nbytes / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}

FORCE_INLINE void write_bcast_scalar(uint32_t cb_id, float value) {
    union {
        float fp32;
        uint32_t bits;
    } scalar = {.fp32 = value};

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);
    auto* write_ptr = reinterpret_cast<tt_l1_ptr uint16_t*>(write_addr);
    write_ptr[0] = scalar.bits >> 16;
    cb_push_back(cb_id, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(0);

    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_partial = 25;
    constexpr uint32_t cb_remote_partials = 26;
    constexpr uint32_t cb_unit_scaler = 27;
    constexpr uint32_t cb_mean_scaler = 29;
    constexpr uint32_t cb_eps = 30;

    const uint32_t total_shard_tiles = get_arg_val<uint32_t>(0);
    const uint32_t shard_h_tiles = get_arg_val<uint32_t>(1);
    const uint32_t core_col = get_arg_val<uint32_t>(3);
    const uint32_t num_cols = get_arg_val<uint32_t>(4);
    const uint32_t semaphore_id = get_arg_val<uint32_t>(5);
    const uint32_t total_width = get_arg_val<uint32_t>(6);

    union {
        uint32_t bits;
        float value;
    } epsilon = {.bits = epsilon_bits};

    experimental::CircularBuffer input_cb(cb_input);
    experimental::CircularBuffer partial_cb(cb_partial);
    experimental::CircularBuffer remote_partials_cb(cb_remote_partials);

    input_cb.push_back(total_shard_tiles);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_unit_scaler>(1.0f);
    write_bcast_scalar(cb_mean_scaler, 1.0f / static_cast<float>(total_width));
    write_bcast_scalar(cb_eps, epsilon.value);

    const uint32_t semaphore_addr = get_semaphore(semaphore_id);
    auto* semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    const uint32_t tile_nbytes = get_tile_size(cb_partial);

    for (uint32_t row = 0; row < shard_h_tiles; ++row) {
        partial_cb.wait_front(1);
        remote_partials_cb.reserve_back(num_cols);

        const uint32_t remote_base_addr = get_write_ptr(cb_remote_partials);
        const uint32_t partial_addr = get_read_ptr(cb_partial);
        const uint32_t local_slot_addr = remote_base_addr + core_col * tile_nbytes;

        noc_semaphore_set(semaphore_ptr, 0);
        copy_tile_local(partial_addr, local_slot_addr, tile_nbytes);

        uint32_t arg_idx = 7;
        for (uint32_t peer_col = 0; peer_col < num_cols; ++peer_col) {
            const uint32_t peer_x = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t peer_y = get_arg_val<uint32_t>(arg_idx++);
            if (peer_col == core_col) {
                continue;
            }

            const uint32_t remote_slot_addr = remote_base_addr + core_col * tile_nbytes;
            noc_async_write(partial_addr, get_noc_addr(peer_x, peer_y, remote_slot_addr), tile_nbytes);
            noc_semaphore_inc(get_noc_addr(peer_x, peer_y, semaphore_addr), 1);
        }

        noc_async_write_barrier();
        if (num_cols > 1) {
            noc_semaphore_wait(semaphore_ptr, num_cols - 1);
            noc_semaphore_set(semaphore_ptr, 0);
        }

        remote_partials_cb.push_back(num_cols);
        partial_cb.pop_front(1);
    }
}
