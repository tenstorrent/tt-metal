// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

// Read a contiguous channel block from a tiled fused projection.  Every block
// has a self-contained three-token prefix: at a time-tile boundary it comes
// from the preceding projected tile-row, and at T=0 from the corresponding
// byte range in the row-major recurrent state.
void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t worker_Ct = get_compile_time_arg_val(1);
    constexpr uint32_t Pt = get_compile_time_arg_val(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t worker_row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t Mt = get_compile_time_arg_val(5);
    constexpr auto projected_a = TensorAccessorArgs<6>();
    constexpr auto state_a = TensorAccessorArgs<projected_a.next_compile_time_args_offset()>();
    constexpr auto tap0_a = TensorAccessorArgs<state_a.next_compile_time_args_offset()>();
    constexpr auto tap1_a = TensorAccessorArgs<tap0_a.next_compile_time_args_offset()>();
    constexpr auto tap2_a = TensorAccessorArgs<tap1_a.next_compile_time_args_offset()>();
    constexpr auto tap3_a = TensorAccessorArgs<tap2_a.next_compile_time_args_offset()>();
    const uint32_t work_start = get_arg_val<uint32_t>(0);
    const uint32_t mt_count = get_arg_val<uint32_t>(1);
    const uint32_t write_state = get_arg_val<uint32_t>(2);
    const uint32_t projected_addr = get_arg_val<uint32_t>(3);
    const uint32_t state_addr = get_arg_val<uint32_t>(4);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(5);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(6);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(7);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(8);

    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t tile_width = 32;
    const uint32_t channel_block = work_start / Mt;
    const uint32_t mt_start = work_start % Mt;
    const uint32_t channel_tile_offset = channel_block * worker_Ct;
    const uint32_t channel_byte_offset = channel_tile_offset * tile_width * sizeof(uint16_t);
    const auto projected = TensorAccessor(projected_a, projected_addr, tile_bytes);
    const auto state = TensorAccessor(state_a, state_addr, row_bytes);
    const auto tap0 = TensorAccessor(tap0_a, tap0_addr, tile_bytes);
    const auto tap1 = TensorAccessor(tap1_a, tap1_addr, tile_bytes);
    const auto tap2 = TensorAccessor(tap2_a, tap2_addr, tile_bytes);
    const auto tap3 = TensorAccessor(tap3_a, tap3_addr, tile_bytes);
    Noc noc;

    CircularBuffer weights(2);
    weights.reserve_back(4 * worker_Ct);
    for (uint32_t ct = 0; ct < worker_Ct; ++ct) {
        const uint32_t source_tile = channel_tile_offset + ct;
        noc.async_read(tap0, weights, tile_bytes, {.page_id = source_tile}, {.offset_bytes = ct * tile_bytes});
        noc.async_read(
            tap1, weights, tile_bytes, {.page_id = source_tile}, {.offset_bytes = (worker_Ct + ct) * tile_bytes});
        noc.async_read(
            tap2, weights, tile_bytes, {.page_id = source_tile}, {.offset_bytes = (2 * worker_Ct + ct) * tile_bytes});
        noc.async_read(
            tap3, weights, tile_bytes, {.page_id = source_tile}, {.offset_bytes = (3 * worker_Ct + ct) * tile_bytes});
    }
    noc.async_read_barrier();
    weights.push_back(4 * worker_Ct);

    CircularBuffer activation(0);
    CircularBuffer prefix(6);
    CircularBuffer new_state(7);
    CircularBuffer projected_tiles(8);
    CircularBuffer projected_rm(9);
    CircularBuffer previous_tiles(10);
    CircularBuffer previous_rm(11);
    prefix.reserve_back(3);
    const uint32_t prefix_ptr = prefix.get_write_ptr();
    UnicastEndpoint self_ep;

    for (uint32_t item = 0; item < mt_count; ++item) {
        const uint32_t mt = mt_start + item;
        if (item == 0) {
            if (mt == 0) {
                for (uint32_t row = 0; row < 3; ++row) {
                    noc.async_read(
                        state,
                        prefix,
                        worker_row_bytes,
                        {.page_id = row, .offset_bytes = channel_byte_offset},
                        {.offset_bytes = row * worker_row_bytes});
                }
            } else {
                previous_tiles.reserve_back(worker_Ct);
                for (uint32_t ct = 0; ct < worker_Ct; ++ct) {
                    noc.async_read(
                        projected,
                        previous_tiles,
                        tile_bytes,
                        {.page_id = (mt - 1) * Pt + channel_tile_offset + ct},
                        {.offset_bytes = ct * tile_bytes});
                }
                noc.async_read_barrier();
                previous_tiles.push_back(worker_Ct);
            }
            if (mt == 0) {
                noc.async_read_barrier();
            }
        }

        projected_tiles.reserve_back(worker_Ct);
        for (uint32_t ct = 0; ct < worker_Ct; ++ct) {
            noc.async_read(
                projected,
                projected_tiles,
                tile_bytes,
                {.page_id = mt * Pt + channel_tile_offset + ct},
                {.offset_bytes = ct * tile_bytes});
        }
        noc.async_read_barrier();
        projected_tiles.push_back(worker_Ct);

        // The compute kernel untilizes this preceding tile into CB 11 before
        // it starts the current tile.  Copying those last rows preserves the
        // exact generic-untilize layout rather than decoding packed faces.
        if (item == 0 && mt != 0) {
            previous_rm.wait_front(worker_Ct);
            const uint32_t previous_rm_ptr = previous_rm.get_read_ptr();
            for (uint32_t row = 0; row < 3; ++row) {
                noc.async_read(
                    self_ep,
                    prefix,
                    worker_row_bytes,
                    {.noc_x = my_x[0], .noc_y = my_y[0], .addr = previous_rm_ptr + (29 + row) * worker_row_bytes},
                    {.offset_bytes = row * worker_row_bytes});
            }
            noc.async_read_barrier();
            previous_rm.pop_front(worker_Ct);
        }

        projected_rm.wait_front(worker_Ct);
        const uint32_t projected_rm_ptr = projected_rm.get_read_ptr();
        for (uint32_t tap = 0; tap < 4; ++tap) {
            activation.reserve_back(worker_Ct);
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t window_row = tap + row;
                const uint32_t source_addr = window_row < 3 ? prefix_ptr + window_row * worker_row_bytes
                                                            : projected_rm_ptr + (window_row - 3) * worker_row_bytes;
                noc.async_read(
                    self_ep,
                    activation,
                    worker_row_bytes,
                    {.noc_x = my_x[0], .noc_y = my_y[0], .addr = source_addr},
                    {.offset_bytes = row * worker_row_bytes});
            }
            noc.async_read_barrier();
            activation.push_back(worker_Ct);
        }

        if (item + 1 < mt_count) {
            for (uint32_t row = 0; row < 3; ++row) {
                noc.async_read(
                    self_ep,
                    prefix,
                    worker_row_bytes,
                    {.noc_x = my_x[0], .noc_y = my_y[0], .addr = projected_rm_ptr + (29 + row) * worker_row_bytes},
                    {.offset_bytes = row * worker_row_bytes});
            }
            noc.async_read_barrier();
        }
        if (write_state && item + 1 == mt_count) {
            new_state.reserve_back(3);
            for (uint32_t row = 0; row < 3; ++row) {
                noc.async_read(
                    self_ep,
                    new_state,
                    worker_row_bytes,
                    {.noc_x = my_x[0], .noc_y = my_y[0], .addr = projected_rm_ptr + (29 + row) * worker_row_bytes},
                    {.offset_bytes = row * worker_row_bytes});
            }
            noc.async_read_barrier();
            new_state.push_back(3);
        }
        projected_rm.pop_front(worker_Ct);
    }
}
