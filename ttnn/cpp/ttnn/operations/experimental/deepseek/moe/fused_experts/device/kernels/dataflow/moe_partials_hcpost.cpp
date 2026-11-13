// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tensor/tensor_accessor.h"

// persistent_moe_request with the hyper-connection post step fused, on one stream-collapse core: this core's
// part of the row is its columns of the width-sharded streams, and its new streams are
//     new[h] = post[h] * x + sum_g comb[g, h] * old[g],     x = the sum of the op's chips' partials of the part
// (compute/moe_partials_hcpost.cpp). This kernel
//   1. reads post and comb (one 32x32 bf16 tile each; post[h] is element (h, 0), comb[g, h] element (g, h), all
//      in face 0) and makes a tile of each value, every element that value: cb_post tile h, cb_comb tile
//      g * num_streams + h;
//   2. copies this core's shard of the old streams (num_streams rows of part_bytes) into cb_s;
//   3. once every chip's flag is set, publishes the partials slots (cb_slots, over the partials shard);
//   4. writes the new streams (cb_out, row h at h * part_bytes) into this core's shard of the output, and clears
//      the flags: the next request, which is the next time the op sends here, comes after this op has finished.
//
// Compile-time args:
//   0: num_slots (the op's chips)   1: tiles (per row of a part)   2: num_streams   3: part_bytes
//   4: cb_slots   5: cb_s   6: cb_post   7: cb_comb   8: cb_out   9: cb_tmp (post's and comb's face-0 heads)
//   10: scalar tile bytes   11+: TensorAccessorArgs(post), then TensorAccessorArgs(comb)
//
// Runtime args:
//   0: skip (a stop request, which gets no partials)   1: partials buffer address   2: post address
//   3: comb address   4: old streams address   5: new streams address (both this core's shard)

namespace {
constexpr uint32_t kFaceHeadBytes = 128;  // face 0's first 4 rows of 16 bf16: every (r < 4, c < 16) element

// A bf16 tile of `bytes` bytes, every element `value`: the compute's elementwise multiply then needs no broadcast.
template <uint32_t bytes>
FORCE_INLINE void fill_tile(uint32_t l1, uint16_t value) {
    const uint32_t pair = (static_cast<uint32_t>(value) << 16) | value;
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < bytes / 4u; ++i) {
        words[i] = pair;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t num_slots = get_compile_time_arg_val(0);
    constexpr uint32_t tiles = get_compile_time_arg_val(1);
    constexpr uint32_t streams = get_compile_time_arg_val(2);
    constexpr uint32_t part_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(4);
    constexpr uint32_t cb_s = get_compile_time_arg_val(5);
    constexpr uint32_t cb_post = get_compile_time_arg_val(6);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(9);
    constexpr uint32_t scalar_tile_bytes = get_compile_time_arg_val(10);
    constexpr auto post_args = TensorAccessorArgs<11>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr uint32_t row_tiles = streams * tiles;
    static_assert(streams <= 4, "post and comb are read from face 0's first 4 rows");

    if (get_arg_val<uint32_t>(0) != 0u) {
        return;
    }
    const uint32_t partials_addr = get_arg_val<uint32_t>(1);
    const auto post = TensorAccessor(post_args, get_arg_val<uint32_t>(2));
    const auto comb = TensorAccessor(comb_args, get_arg_val<uint32_t>(3));
    const uint32_t old_addr = get_arg_val<uint32_t>(4);
    const uint32_t new_addr = get_arg_val<uint32_t>(5);
    auto* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(partials_addr + num_slots * part_bytes);

    // 1. and 2.
    const uint32_t tmp = get_write_ptr(cb_tmp);
    noc_async_read(post.get_noc_addr(0), tmp, kFaceHeadBytes);
    noc_async_read(comb.get_noc_addr(0), tmp + kFaceHeadBytes, kFaceHeadBytes);
    cb_reserve_back(cb_s, row_tiles);
    noc_async_read(get_noc_addr(old_addr), get_write_ptr(cb_s), streams * part_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_s, row_tiles);

    auto* head = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp);
    cb_reserve_back(cb_comb, streams * streams);
    const uint32_t comb_l1 = get_write_ptr(cb_comb);
    for (uint32_t g = 0; g < streams; ++g) {
        for (uint32_t h = 0; h < streams; ++h) {
            fill_tile<scalar_tile_bytes>(
                comb_l1 + (g * streams + h) * scalar_tile_bytes, head[kFaceHeadBytes / 2u + g * 16u + h]);
        }
    }
    cb_push_back(cb_comb, streams * streams);
    cb_reserve_back(cb_post, streams);
    const uint32_t post_l1 = get_write_ptr(cb_post);
    for (uint32_t h = 0; h < streams; ++h) {
        fill_tile<scalar_tile_bytes>(post_l1 + h * scalar_tile_bytes, head[h * 16u]);
    }
    cb_push_back(cb_post, streams);

    // 3.
    {
        DeviceZoneScopedN("MOE_SUM_WAIT");
        for (uint32_t slot = 0; slot < num_slots; ++slot) {
            do {
                invalidate_l1_cache();
            } while (flags[slot] == 0u);
        }
    }
    cb_push_back(cb_slots, num_slots * tiles);

    // 4.
    {
        DeviceZoneScopedN("MOE_SUM_WRITE");
        cb_wait_front(cb_out, row_tiles);
        noc_async_write(get_read_ptr(cb_out), get_noc_addr(new_addr), streams * part_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, row_tiles);
    }
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        flags[slot] = 0u;
    }
}
