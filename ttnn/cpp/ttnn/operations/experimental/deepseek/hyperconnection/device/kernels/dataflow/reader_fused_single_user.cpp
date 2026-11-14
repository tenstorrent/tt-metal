// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "collapsed_row_scatter.h"

namespace {

FORCE_INLINE uint32_t tile_face_index(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
    return face * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace

// The same reader is compiled three times with a role:
//   0: collapse cores [0,7]
//   1: post core 8
//   2: comb core 9
//
// fused_w is a one-core width shard on core (0, 0); every core that needs it reads its first row
// straight out of that core's L1. Pushed (the program computes fused_w itself), the fn dot's root
// writes the row into cb_fused_w on every core that needs it and then raises fused_w_sem.
void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fused_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_hidden = get_compile_time_arg_val(2);
    constexpr uint32_t cb_params = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(4);
    constexpr uint32_t cb_pre = get_compile_time_arg_val(5);
    constexpr uint32_t num_streams = get_compile_time_arg_val(6);
    constexpr bool fused_w_row_major = get_compile_time_arg_val(7) != 0;
    constexpr bool use_pre_mix = get_compile_time_arg_val(8) != 0;
    constexpr bool hidden_row_major = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t hidden_pages = get_compile_time_arg_val(10);
    constexpr uint32_t fused_w_noc_x = get_compile_time_arg_val(11);
    constexpr uint32_t fused_w_noc_y = get_compile_time_arg_val(12);
    // comb_bias as [1,1,1,H*H]: its H*H <= 16 values are row 0 of face 0, not an H x H block.
    constexpr bool comb_bias_flat = get_compile_time_arg_val(13) != 0;
    // Role 0, ROW_MAJOR streams: the collapsed row, in row_chunks pages of cb_collapsed_row, and the
    // out_tiles-tile output shard behind cb_collapsed_out, whose second half this RISC writes.
    constexpr uint32_t cb_collapsed_row = get_compile_time_arg_val(14);
    constexpr uint32_t row_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t cb_collapsed_out = get_compile_time_arg_val(16);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(17);
    // A ROW_MAJOR collapsed output: the compute packs the row into the output shard, nothing to scatter.
    constexpr bool collapsed_row_major = get_compile_time_arg_val(18) != 0;
    constexpr bool fused_w_pushed = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t fused_w_sem = get_compile_time_arg_val(20);

    constexpr auto pre_bias_args = TensorAccessorArgs<21>();
    constexpr auto post_bias_args = TensorAccessorArgs<pre_bias_args.next_compile_time_args_offset()>();
    constexpr auto comb_bias_args = TensorAccessorArgs<post_bias_args.next_compile_time_args_offset()>();
    constexpr auto pre_mix_args = TensorAccessorArgs<comb_bias_args.next_compile_time_args_offset()>();

    const uint32_t fused_w_addr = get_arg_val<uint32_t>(0);
    const uint32_t pre_bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_bias_addr = get_arg_val<uint32_t>(3);
    const uint32_t pre_mix_addr = get_arg_val<uint32_t>(4);
    const uint32_t collapsed_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t one_tile = 1;
    // Row 0 of the bias tiles (row 0 of face 0) and rows 0..H-1 of the comb bias, as read from DRAM.
    constexpr uint32_t bias_row_bytes = 64;
    constexpr uint32_t comb_bias_bytes = comb_bias_flat ? bias_row_bytes : num_streams * 16 * sizeof(uint16_t);
    // Where the post core stages its pre bias row inside cb_stage.
    constexpr uint32_t pre_bias_stage_offset = 1024;

    Noc noc;
    CircularBuffer cb_fw(cb_fused_w);

    // fused_w's first row: a ROW_MAJOR row, or row 0 of faces 0 and 1 of a tile. A pushed row is
    // only waited for (wait_fused_w), once everything that does not need it is under way.
    auto read_fused_w = [&]() {
        cb_fw.reserve_back(one_tile);
        if constexpr (fused_w_pushed) {
            return;
        }
        const uint32_t l1 = cb_fw.get_write_ptr();
        if constexpr (fused_w_row_major) {
            noc_async_read(get_noc_addr(fused_w_noc_x, fused_w_noc_y, fused_w_addr), l1, 64);
        } else {
            noc_async_read(get_noc_addr(fused_w_noc_x, fused_w_noc_y, fused_w_addr), l1, 32);
            noc_async_read(get_noc_addr(fused_w_noc_x, fused_w_noc_y, fused_w_addr + 512), l1 + 512, 32);
        }
    };
    auto wait_fused_w = [&]() {
        if constexpr (fused_w_pushed) {
            Semaphore<> landed(fused_w_sem);
            landed.wait(1);
        }
    };
    const volatile tt_l1_ptr uint16_t* fw = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_fw.get_write_ptr());
    auto fused_w_at = [&](uint32_t k) {
        if constexpr (fused_w_row_major) {
            return fw[k];
        } else {
            return fw[tile_face_index(0, k & 31u)];
        }
    };

    // The collapse computes its own pre unless given pre_mix; the post core computes post, and pre
    // to return it when the collapse took pre_mix instead.
    constexpr bool with_post = role == 1;
    constexpr bool with_pre = (role == 0 && !use_pre_mix) || (role == 1 && use_pre_mix);

    if constexpr (role == 0) {
        // The hidden shard is consumed in place: 32x32 tiles, or -- ROW_MAJOR -- each stream's row
        // read as consecutive h x 32 tiles, whose faces are exactly consecutive runs of the row.
        CircularBuffer cb_h(cb_hidden);
        cb_h.reserve_back(hidden_pages);
        cb_h.push_back(hidden_pages);

        if constexpr (use_pre_mix) {
            const auto pre_mix = TensorAccessor(pre_mix_args, pre_mix_addr);
            CircularBuffer cb_p(cb_pre);
            cb_p.reserve_back(one_tile);
            // The row-major collapse reads only the H weights; the tiled one is a matmul on the tile.
            constexpr uint32_t pre_mix_bytes = hidden_row_major ? 64 : 2048;
            noc.async_read(pre_mix, cb_p, pre_mix_bytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_p.push_back(one_tile);
        }
    }

    if constexpr (with_post || with_pre || role == 2) {
        // One parameter tile the compute turns into the outputs (see hc_single_user_sfpu.h); zeroed
        // except for comb, which reads only its H x H block.
        CircularBuffer cb_prm(cb_params);
        CircularBuffer cb_stg(cb_stage);
        cb_prm.reserve_back(one_tile);
        cb_stg.reserve_back(one_tile);
        if constexpr (role != 2) {
            noc.async_write_zeros(cb_prm, cb_prm.get_tile_size(), {.offset_bytes = 0});
        }
        read_fused_w();
        if constexpr (role != 2) {
            if constexpr (with_post) {
                const auto post_bias = TensorAccessor(post_bias_args, post_bias_addr);
                noc.async_read(post_bias, cb_stg, bias_row_bytes, {.page_id = 0}, {.offset_bytes = 0});
            }
            if constexpr (with_pre) {
                const auto pre_bias = TensorAccessor(pre_bias_args, pre_bias_addr);
                noc.async_read(
                    pre_bias, cb_stg, bias_row_bytes, {.page_id = 0}, {.offset_bytes = pre_bias_stage_offset});
            }
        } else {
            const auto comb_bias = TensorAccessor(comb_bias_args, comb_bias_addr);
            noc.async_read(comb_bias, cb_stg, comb_bias_bytes, {.page_id = 0}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        if constexpr (role != 2) {
            noc.write_zeros_l1_barrier();
        }
        wait_fused_w();

        auto* prm = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_prm.get_write_ptr());
        const auto* stg = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_stg.get_write_ptr());
        if constexpr (role == 0 && hidden_row_major) {
            // An h x 32 tile like the hidden ones: row 0 interleaves pre_w[k] and pre_b[k].
            for (uint32_t k = 0; k < num_streams; ++k) {
                prm[2 * k] = fused_w_at(k);
                prm[2 * k + 1] = stg[pre_bias_stage_offset / sizeof(uint16_t) + k];
            }
        } else if constexpr (role != 2) {
            // Faces 0 / 1: post_w / post_b down column 0. Faces 2 / 3: pre_w / pre_b along row 0.
            for (uint32_t k = 0; k < num_streams; ++k) {
                if constexpr (with_post) {
                    prm[tile_face_index(k, 0)] = fused_w_at(num_streams + k);
                    prm[tile_face_index(k, 16)] = stg[k];
                }
                if constexpr (with_pre) {
                    prm[tile_face_index(16, k)] = fused_w_at(k);
                    prm[tile_face_index(16, 16 + k)] = stg[pre_bias_stage_offset / sizeof(uint16_t) + k];
                }
            }
        } else {
            // comb_w in face 0 and comb_b in face 1, with (r, c) at face row r + 4 * (c / 2), face
            // column c % 2 (see hc_comb_sinkhorn).
            constexpr uint32_t comb_base = 2u * num_streams;
            auto comb_bias_at = [&](uint32_t r, uint32_t c) {
                return comb_bias_flat ? r * num_streams + c : tile_face_index(r, c);
            };
            if constexpr (num_streams % 2 == 0) {
                // Columns 2h and 2h + 1 of a row are adjacent, and word-aligned, on both sides.
                auto* prm32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(prm);
                const auto* fw32 = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(fw);
                const auto* stg32 = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(stg);
#pragma GCC unroll 8
                for (uint32_t i = 0; i < num_streams * num_streams / 2; ++i) {
                    const uint32_t r = i / (num_streams / 2);
                    const uint32_t c = 2 * (i % (num_streams / 2));
                    const uint32_t fw_k = comb_base + r * num_streams + c;
                    const uint32_t fr = r + 2 * c;
                    prm32[tile_face_index(fr, 0) / 2] = fw32[(fused_w_row_major ? fw_k : tile_face_index(0, fw_k)) / 2];
                    prm32[tile_face_index(fr, 16) / 2] = stg32[comb_bias_at(r, c) / 2];
                }
            } else {
                for (uint32_t r = 0; r < num_streams; ++r) {
                    for (uint32_t c = 0; c < num_streams; ++c) {
                        const uint32_t fr = r + 4 * (c >> 1);
                        prm[tile_face_index(fr, c & 1)] = fused_w_at(comb_base + r * num_streams + c);
                        prm[tile_face_index(fr, 16 + (c & 1))] = stg[comb_bias_at(r, c)];
                    }
                }
            }
        }
        cb_prm.push_back(one_tile);
    }

    if constexpr (role == 0 && hidden_row_major && !collapsed_row_major) {
        scatter_collapsed_row(
            cb_collapsed_row, row_chunks, cb_collapsed_out, collapsed_addr, out_tiles / 2, out_tiles - out_tiles / 2);
    }
}
