// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/select_final_carry/device/kernels/dataflow/select_final_carry_common.hpp"

constexpr uint32_t tiles_cb = get_compile_time_arg_val(0);
constexpr uint32_t mode_cb = get_compile_time_arg_val(1);
constexpr uint32_t packet_pages = get_compile_time_arg_val(2);
constexpr uint32_t sp_rank = get_compile_time_arg_val(3);
constexpr uint32_t sp_size = get_compile_time_arg_val(4);
constexpr uint32_t local_rows = get_compile_time_arg_val(5);
constexpr bool has_actual_end = get_compile_time_arg_val(6);
// The local final is the last of `tail_groups` group states stored per head, each `head_tiles` tiles.
constexpr uint32_t tail_groups = get_compile_time_arg_val(7);
constexpr uint32_t head_tiles = get_compile_time_arg_val(8);
constexpr auto tail_args = TensorAccessorArgs<9>();
constexpr auto prefix_args = TensorAccessorArgs<tail_args.next_compile_time_args_offset()>();
constexpr auto start_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
constexpr auto end_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();

template <bool last_group, typename Accessor>
void stream_tiles(const Accessor& source, uint32_t tile, uint32_t end_tile) {
    const uint32_t page_bytes = get_tile_size(tiles_cb);
    while (tile < end_tile) {
        cb_reserve_back(tiles_cb, packet_pages);
        uint32_t l1_address = get_write_ptr(tiles_cb);
        const uint32_t packet_end = tile + packet_pages < end_tile ? tile + packet_pages : end_tile;
        for (; tile < packet_end; ++tile, l1_address += page_bytes) {
            const uint32_t page =
                last_group ? (tile / head_tiles * tail_groups + tail_groups - 1) * head_tiles + tile % head_tiles
                           : tile;
            noc_async_read_page(page, source, l1_address);
        }
        noc_async_read_barrier();
        cb_push_back(tiles_cb, packet_pages);
    }
}

// Derive the chronology, publish this rank's mode to the writer, and stream the tiles it sends or copies.
void kernel_main() {
    using namespace kda_select_final_carry;
    size_t arg = 0;
    const uint32_t tail_address = get_arg_val<uint32_t>(arg++);
    const uint32_t prefix_address = get_arg_val<uint32_t>(arg++);
    const uint32_t start_address = get_arg_val<uint32_t>(arg++);
    const uint32_t end_address = get_arg_val<uint32_t>(arg++);
    const uint32_t copy_begin = get_arg_val<uint32_t>(arg++);
    const uint32_t copy_end = get_arg_val<uint32_t>(arg++);
    const uint32_t send_begin = get_arg_val<uint32_t>(arg++);
    const uint32_t send_end = get_arg_val<uint32_t>(arg++);

    cb_reserve_back(mode_cb, 1);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(mode_cb));
    const auto start = TensorAccessor(start_args, start_address);
    noc_async_read(start.get_noc_addr(0), get_write_ptr(mode_cb), sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t actual_start = words[0];
    kda_chronology::Topology topology{};
    if constexpr (has_actual_end) {
        const auto end = TensorAccessor(end_args, end_address);
        noc_async_read(end.get_noc_addr(0), get_write_ptr(mode_cb), sizeof(uint32_t));
        noc_async_read_barrier();
        topology = kda_chronology::derive_interval(actual_start, words[0], sp_rank, sp_size, local_rows);
    } else {
        topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    }
    const Mode mode = !topology.split ? copy_prefix : (topology.final_owner == sp_rank ? broadcast : receive);
    words[0] = mode;
    cb_push_back(mode_cb, 1);

    if (mode == copy_prefix) {
        stream_tiles<false>(TensorAccessor(prefix_args, prefix_address), copy_begin, copy_end);
    } else if (mode == broadcast) {
        stream_tiles<true>(TensorAccessor(tail_args, tail_address), send_begin, send_end);
    }
}
