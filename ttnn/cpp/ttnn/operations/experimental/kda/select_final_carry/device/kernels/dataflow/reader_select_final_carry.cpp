// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/select_final_carry/device/kernels/dataflow/select_final_carry_common.hpp"

constexpr uint32_t tiles_cb = get_compile_time_arg_val(0);
constexpr uint32_t mode_cb = get_compile_time_arg_val(1);
constexpr uint32_t packet_pages = get_compile_time_arg_val(2);
constexpr uint32_t sp_size = get_compile_time_arg_val(3);
constexpr uint32_t local_rows = get_compile_time_arg_val(4);
constexpr bool has_actual_end = get_compile_time_arg_val(5);
// The local final is the last of `tail_groups` group states stored per head, each `head_tiles` tiles.
constexpr uint32_t tail_groups = get_compile_time_arg_val(6);
constexpr uint32_t head_tiles = get_compile_time_arg_val(7);
// Slab (slab_kernel_args): the output is entry `index * slab_scale + slab_offset`; none past slab_index_limit.
constexpr bool slab = get_compile_time_arg_val(8);
constexpr uint32_t slab_index_limit = get_compile_time_arg_val(9);
constexpr uint32_t slab_scale = get_compile_time_arg_val(10);
constexpr uint32_t slab_offset = get_compile_time_arg_val(11);
constexpr uint32_t entry_tiles = get_compile_time_arg_val(12);
// mode_cb's second aligned slot, where the slab index lands.
constexpr uint32_t slab_index_word = get_compile_time_arg_val(13);
constexpr auto tail_args = TensorAccessorArgs<14>();
constexpr auto prefix_args = TensorAccessorArgs<tail_args.next_compile_time_args_offset()>();
constexpr auto start_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
constexpr auto end_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();
constexpr auto slab_index_args = TensorAccessorArgs<end_args.next_compile_time_args_offset()>();

// Packets whose reads are issued under one barrier: deep enough to keep a link busy, so DRAM read latency is paid once
// per batch instead of once per packet. The tiles buffer holds two batches.
constexpr uint32_t batch_packets = 8;

template <bool last_group, typename Accessor>
void stream_tiles(const Accessor& source, uint32_t tile, uint32_t end_tile) {
    const uint32_t page_bytes = get_tile_size(tiles_cb);
    while (tile < end_tile) {
        const uint32_t remaining_packets = (end_tile - tile + packet_pages - 1) / packet_pages;
        const uint32_t packets = remaining_packets < batch_packets ? remaining_packets : batch_packets;
        cb_reserve_back(tiles_cb, packets * packet_pages);
        uint32_t l1_address = get_write_ptr(tiles_cb);
        const uint32_t batch_end = tile + packets * packet_pages < end_tile ? tile + packets * packet_pages : end_tile;
        for (; tile < batch_end; ++tile, l1_address += page_bytes) {
            const uint32_t page =
                last_group ? (tile / head_tiles * tail_groups + tail_groups - 1) * head_tiles + tile % head_tiles
                           : tile;
            noc_async_read_page(page, source, l1_address);
        }
        noc_async_read_barrier();
        cb_push_back(tiles_cb, packets * packet_pages);
    }
}

// Derive the chronology, publish this rank's mode to the writer, and stream the tiles it sends.
void kernel_main() {
    using namespace kda_select_final_carry;
    const uint32_t tail_address = get_common_arg_val<uint32_t>(0);
    const uint32_t prefix_address = get_common_arg_val<uint32_t>(1);
    const uint32_t start_address = get_common_arg_val<uint32_t>(2);
    const uint32_t end_address = get_common_arg_val<uint32_t>(3);
    const uint32_t slab_index_address = get_common_arg_val<uint32_t>(4);
    size_t arg = 0;
    const uint32_t send_begin = get_arg_val<uint32_t>(arg++);
    const uint32_t send_end = get_arg_val<uint32_t>(arg++);
    const uint32_t sp_rank = get_arg_val<uint32_t>(arg++);

    cb_reserve_back(mode_cb, 1);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(mode_cb));
    const auto start = TensorAccessor(start_args, start_address);
    noc_async_read(start.get_noc_addr(0), get_write_ptr(mode_cb), sizeof(uint32_t));
    if constexpr (slab) {
        const auto slab_index = TensorAccessor(slab_index_args, slab_index_address);
        noc_async_read(
            slab_index.get_noc_addr(0), get_write_ptr(mode_cb) + slab_index_word * sizeof(uint32_t), sizeof(uint32_t));
    }
    noc_async_read_barrier();
    const uint32_t actual_start = words[0];
    const uint32_t index = slab ? words[slab_index_word] : 0;
    const bool writes = !slab || index < slab_index_limit;
    kda_chronology::Topology topology{};
    if constexpr (has_actual_end) {
        const auto end = TensorAccessor(end_args, end_address);
        noc_async_read(end.get_noc_addr(0), get_write_ptr(mode_cb), sizeof(uint32_t));
        noc_async_read_barrier();
        topology = kda_chronology::derive_interval(actual_start, words[0], sp_rank, sp_size, local_rows);
    } else {
        topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    }
    // The final state is the split tail's, else the prefix's, which only the last chronological rank holds.
    const uint32_t holder = topology.split ? topology.final_owner : (topology.first_rank + sp_size - 1) % sp_size;
    const Mode mode = holder == sp_rank ? broadcast : receive;
    words[mode_word] = mode;
    words[output_base_word] = writes ? (index * slab_scale + slab_offset) * entry_tiles : 0;
    words[write_word] = writes;
    cb_push_back(mode_cb, 1);

    if (mode == broadcast && writes) {
        if (topology.split) {
            stream_tiles<true>(TensorAccessor(tail_args, tail_address), send_begin, send_end);
        } else {
            stream_tiles<false>(TensorAccessor(prefix_args, prefix_address), send_begin, send_end);
        }
    }
}
