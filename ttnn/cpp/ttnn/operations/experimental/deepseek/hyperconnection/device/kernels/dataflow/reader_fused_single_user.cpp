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
//
// With the folded mix step a collapse core also reads the mix's inputs -- mix_comb's and mix_post's H rows, its
// D/8 columns of mix_sublayer_out -- and lays the weights out for the compute (see hc_mix_collapse_streams). Once the
// compute has packed the mixed shard it raises mix_sem on the fn dot cores that read their K-slices out of it.
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
    constexpr bool use_mix = get_compile_time_arg_val(21) != 0;
    constexpr uint32_t cb_mix_stage = get_compile_time_arg_val(22);
    constexpr uint32_t cb_mix_w = get_compile_time_arg_val(23);
    constexpr uint32_t cb_mix_sub = get_compile_time_arg_val(24);
    constexpr uint32_t cb_mixed = get_compile_time_arg_val(25);
    constexpr uint32_t sub_elems_per_page = get_compile_time_arg_val(26);
    constexpr uint32_t mix_sem = get_compile_time_arg_val(27);
    // The fused collapsed RMSNorm: role 0 reads its row_bytes of the gain into cb_norm_gamma.
    constexpr bool fuse_norm = get_compile_time_arg_val(28) != 0;
    constexpr uint32_t cb_norm_gamma = get_compile_time_arg_val(29);
    constexpr uint32_t norm_row_bytes = get_compile_time_arg_val(30);
    // A block of rows > 1 rows (FusedSingleUserParams): see the rows branches below.
    constexpr uint32_t rows = get_compile_time_arg_val(31);
    [[maybe_unused]] constexpr uint32_t num_w_tiles = get_compile_time_arg_val(32);
    [[maybe_unused]] constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(33);

    constexpr auto pre_bias_args = TensorAccessorArgs<34>();
    constexpr auto post_bias_args = TensorAccessorArgs<pre_bias_args.next_compile_time_args_offset()>();
    constexpr auto comb_bias_args = TensorAccessorArgs<post_bias_args.next_compile_time_args_offset()>();
    constexpr auto pre_mix_args = TensorAccessorArgs<comb_bias_args.next_compile_time_args_offset()>();
    constexpr auto mix_post_args = TensorAccessorArgs<pre_mix_args.next_compile_time_args_offset()>();
    constexpr auto mix_comb_args = TensorAccessorArgs<mix_post_args.next_compile_time_args_offset()>();
    constexpr auto mix_sub_args = TensorAccessorArgs<mix_comb_args.next_compile_time_args_offset()>();
    constexpr auto norm_gamma_args = TensorAccessorArgs<mix_sub_args.next_compile_time_args_offset()>();

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

    if constexpr (role == 0 && rows > 1) {
        // A block: the streams, the sublayer output, collapsed and the gain are (rows x 32)-tiled, read as the
        // compute's virtual tiles (hc_rows_sfpu.h). Each row has its own pre_mix / mix_comb / mix_post tile; the
        // compute takes them as per-row weight vectors (vector v's sub-row lr holds its weight for the block row
        // that sub-row is), laid out here in cb_mix_w's 32x32 tiles.
        CircularBuffer cb_h(cb_hidden);
        cb_h.reserve_back(hidden_pages);
        cb_h.push_back(hidden_pages);

        // Bytes [start, start + num_bytes) of a tensor read as one flat byte stream of page_bytes pages.
        auto read_flat =
            [&](const auto& acc, uint32_t page_bytes, uint32_t start, uint32_t num_bytes, CircularBuffer& cb) {
                uint32_t pos = start;
                uint32_t dst = 0;
                while (pos < start + num_bytes) {
                    const uint32_t in_page = pos % page_bytes;
                    uint32_t n = page_bytes - in_page;
                    if (n > start + num_bytes - pos) {
                        n = start + num_bytes - pos;
                    }
                    noc.async_read(
                        acc, cb, n, {.page_id = pos / page_bytes, .offset_bytes = in_page}, {.offset_bytes = dst});
                    pos += n;
                    dst += n;
                }
            };
        CircularBuffer cb_g(cb_norm_gamma);
        if constexpr (fuse_norm) {
            const auto gamma = TensorAccessor(norm_gamma_args, get_arg_val<uint32_t>(6));
            cb_g.reserve_back(row_chunks);
            read_flat(gamma, gamma_page_bytes, get_arg_val<uint32_t>(7), norm_row_bytes, cb_g);
        }

        // Stage: pre_mix row j at byte 64 j; mix_comb / mix_post tile j's first H face rows at 512 / 1536 + 128 j.
        constexpr uint32_t comb_stage = 512;
        constexpr uint32_t post_stage = 1536;
        constexpr uint32_t face_rows_bytes = num_streams * 16 * sizeof(uint16_t);
        static_assert(
            rows == 1 || (face_rows_bytes <= 128 && rows <= 8), "a block's stage holds 8 rows of at most 4 streams");
        CircularBuffer cb_stg(cb_mix_stage);
        CircularBuffer cb_w(cb_mix_w);
        CircularBuffer cb_s(cb_mix_sub);
        cb_stg.reserve_back(2);
        cb_w.reserve_back(num_w_tiles);
        const auto pre_mix = TensorAccessor(pre_mix_args, pre_mix_addr);
        for (uint32_t j = 0; j < rows; ++j) {
            noc.async_read(pre_mix, cb_stg, 64, {.page_id = j}, {.offset_bytes = 64 * j});
        }
        if constexpr (use_mix) {
            const auto mix_post = TensorAccessor(mix_post_args, get_arg_val<uint32_t>(8));
            const auto mix_comb = TensorAccessor(mix_comb_args, get_arg_val<uint32_t>(9));
            const auto mix_sub = TensorAccessor(mix_sub_args, get_arg_val<uint32_t>(10));
            const uint32_t col_start = get_arg_val<uint32_t>(11);
            for (uint32_t j = 0; j < rows; ++j) {
                noc.async_read(
                    mix_comb, cb_stg, face_rows_bytes, {.page_id = j}, {.offset_bytes = comb_stage + 128 * j});
                noc.async_read(
                    mix_post, cb_stg, face_rows_bytes, {.page_id = j}, {.offset_bytes = post_stage + 128 * j});
            }
            // This core's columns of every row: in the block's byte order, out_tiles (rows x 32) tiles from tile
            // col_start / 32 on.
            cb_s.reserve_back(row_chunks);
            constexpr uint32_t row_tile_bytes = rows * 32 * sizeof(uint16_t);
            read_flat(
                mix_sub,
                sub_elems_per_page * sizeof(uint16_t),
                (col_start / 32) * row_tile_bytes,
                out_tiles * row_tile_bytes,
                cb_s);
        }
        noc.async_read_barrier();

        const auto* stg = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_stg.get_write_ptr());
        auto pre_at = [&](uint32_t j, uint32_t r) -> uint32_t { return stg[32 * j + r]; };
        constexpr uint32_t num_q = use_mix ? num_streams * (num_streams + 1) + num_streams : num_streams;
        auto weight = [&](uint32_t q, uint32_t j) -> uint32_t {
            if constexpr (!use_mix) {
                return pre_at(j, q);
            } else {
                constexpr uint32_t mix_q = num_streams * (num_streams + 1);
                if (q >= mix_q) {
                    return pre_at(j, q - mix_q);
                }
                const uint32_t r = q / (num_streams + 1);
                const uint32_t k = q % (num_streams + 1);
                if (k < num_streams) {
                    // comb[k][r]: tile j, face row k, column r.
                    return stg[(comb_stage + 128 * j) / 2 + 16 * k + r];
                }
                // post[r]: tile j, face row r, column 0.
                return stg[(post_stage + 128 * j) / 2 + 16 * r];
            }
        };
        constexpr uint32_t G = rows > 4 ? 2 : 1;
        constexpr uint32_t num_vectors = num_q * G;
        static_assert(rows == 1 || num_vectors <= 32 * num_w_tiles, "the per-row weights overflow cb_mix_w");
        auto* w32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_w.get_write_ptr());
        // Vectors v and v + 1 share a face and its four rows (columns of parity 0 / 1): one word per column pair.
        for (uint32_t v = 0; v < num_vectors; v += 2) {
            const uint32_t i = v % 32;
            const uint32_t base = (v / 32) * 512 + (i / 8) * 128 + ((i % 8) / 2) * 4 * 8;
            for (uint32_t lr = 0; lr < 4; ++lr) {
                const uint32_t j0 = G == 2 ? 4 * (v % G) + lr : lr % rows;
                const uint32_t j1 = G == 2 ? 4 * ((v + 1) % G) + lr : lr % rows;
                const uint32_t lo = weight(v / G, j0);
                const uint32_t hi = v + 1 < num_vectors ? weight((v + 1) / G, j1) : 0u;
                const uint32_t word = lo | (hi << 16);
                volatile tt_l1_ptr uint32_t* dst = w32 + base + lr * 8;
#pragma GCC unroll 8
                for (uint32_t cc = 0; cc < 8; ++cc) {
                    dst[cc] = word;
                }
            }
        }
        cb_w.push_back(num_w_tiles);
        if constexpr (use_mix) {
            cb_s.push_back(row_chunks);
        }
        if constexpr (fuse_norm) {
            cb_g.push_back(row_chunks);
        }
        if constexpr (use_mix) {
            // The fn dot reads the mixed shard as soon as it is packed.
            CircularBuffer cb_m(cb_mixed);
            cb_m.wait_front(hidden_pages);
            const uint32_t num_dests = get_arg_val<uint32_t>(12);
            Semaphore<> mixed_ready(mix_sem);
            for (uint32_t i = 0; i < num_dests; ++i) {
                mixed_ready.up(noc, get_arg_val<uint32_t>(13 + 2 * i), get_arg_val<uint32_t>(14 + 2 * i), 1);
            }
            noc.async_atomic_barrier();
        }
    } else if constexpr (role == 0) {
        // The hidden shard is consumed in place: 32x32 tiles, or -- ROW_MAJOR -- each stream's row
        // read as consecutive h x 32 tiles, whose faces are exactly consecutive runs of the row.
        CircularBuffer cb_h(cb_hidden);
        cb_h.reserve_back(hidden_pages);
        cb_h.push_back(hidden_pages);

        // The norm's gain: this core's columns of the [1, D] row, laid out as the collapsed row's h x 32 tiles.
        if constexpr (fuse_norm) {
            const auto gamma = TensorAccessor(norm_gamma_args, get_arg_val<uint32_t>(6));
            CircularBuffer cb_g(cb_norm_gamma);
            cb_g.reserve_back(row_chunks);
            noc.async_read(
                gamma,
                cb_g,
                norm_row_bytes,
                {.page_id = 0, .offset_bytes = get_arg_val<uint32_t>(7)},
                {.offset_bytes = 0});
        }

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

        if constexpr (use_mix) {
            const auto mix_post = TensorAccessor(mix_post_args, get_arg_val<uint32_t>(8));
            const auto mix_comb = TensorAccessor(mix_comb_args, get_arg_val<uint32_t>(9));
            const auto mix_sub = TensorAccessor(mix_sub_args, get_arg_val<uint32_t>(10));
            const uint32_t col_start = get_arg_val<uint32_t>(11);
            CircularBuffer cb_stg(cb_mix_stage);
            CircularBuffer cb_w(cb_mix_w);
            CircularBuffer cb_s(cb_mix_sub);
            cb_stg.reserve_back(one_tile);
            cb_w.reserve_back(one_tile);
            cb_s.reserve_back(row_chunks);
            // Rows 0..H-1 of face 0 of both tiles: comb[r][c] at element 16r + c, post[r] at element 16r.
            constexpr uint32_t face_rows_bytes = num_streams * 16 * sizeof(uint16_t);
            constexpr uint32_t post_stage_offset = 1024;
            noc.async_read(mix_comb, cb_stg, face_rows_bytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read(mix_post, cb_stg, face_rows_bytes, {.page_id = 0}, {.offset_bytes = post_stage_offset});
            // This core's columns [col_start, col_start + out_tiles * 32), a page piece at a time: the core's slice
            // of the row, which the h x 32 tiles of cb_mix_sub hold in order.
            constexpr uint32_t num_cols = out_tiles * 32;
            uint32_t col = col_start;
            uint32_t dst_offset = 0;
            while (col < col_start + num_cols) {
                const uint32_t in_page = col % sub_elems_per_page;
                uint32_t n = sub_elems_per_page - in_page;
                if (n > col_start + num_cols - col) {
                    n = col_start + num_cols - col;
                }
                noc.async_read(
                    mix_sub,
                    cb_s,
                    n * sizeof(uint16_t),
                    {.page_id = col / sub_elems_per_page, .offset_bytes = in_page * sizeof(uint16_t)},
                    {.offset_bytes = dst_offset});
                col += n;
                dst_offset += n * sizeof(uint16_t);
            }
            noc.async_read_barrier();

            // Row r of the weights: comb[0..H-1][r] (column r of comb, which mixes into stream r), then post[r].
            const auto* stg = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_stg.get_write_ptr());
            auto* w = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_w.get_write_ptr());
            for (uint32_t r = 0; r < num_streams; ++r) {
                for (uint32_t j = 0; j < num_streams; ++j) {
                    w[r * (num_streams + 1) + j] = stg[j * 16 + r];
                }
                w[r * (num_streams + 1) + num_streams] = stg[post_stage_offset / sizeof(uint16_t) + r * 16];
            }
            cb_w.push_back(one_tile);
            cb_s.push_back(row_chunks);

            // The fn dot reads the mixed shard as soon as it is packed.
            CircularBuffer cb_m(cb_mixed);
            cb_m.wait_front(hidden_pages);
            const uint32_t num_dests = get_arg_val<uint32_t>(12);
            Semaphore<> mixed_ready(mix_sem);
            for (uint32_t i = 0; i < num_dests; ++i) {
                mixed_ready.up(noc, get_arg_val<uint32_t>(13 + 2 * i), get_arg_val<uint32_t>(14 + 2 * i), 1);
            }
            noc.async_atomic_barrier();
        }
        if constexpr (fuse_norm) {
            noc.async_read_barrier();
            CircularBuffer(cb_norm_gamma).push_back(row_chunks);
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
        // A block's fused_w is one (rows x 32) tile: row j's value q in face q / 16, face row j.
        [[maybe_unused]] auto fw_row = [&](uint32_t j, uint32_t q) -> uint16_t {
            return fw[(q / 16) * rows * 16 + j * 16 + (q % 16)];
        };
        if constexpr (rows > 1 && role == 1) {
            static_assert(rows == 1 || use_pre_mix, "a block's post core also returns pre");
            // Stream k of block row j at (row k, column j) of every face: post_w, post_b, pre_w, pre_b.
            for (uint32_t j = 0; j < rows; ++j) {
                for (uint32_t k = 0; k < num_streams; ++k) {
                    prm[tile_face_index(k, j)] = fw_row(j, num_streams + k);
                    prm[tile_face_index(k, 16 + j)] = stg[k];
                    prm[tile_face_index(16 + k, j)] = fw_row(j, k);
                    prm[tile_face_index(16 + k, 16 + j)] = stg[pre_bias_stage_offset / sizeof(uint16_t) + k];
                }
            }
        } else if constexpr (rows > 1 && role == 2) {
            // comb_w / comb_b (r, c) of block row j at face row r + 4 * (c / 2), face column 2 j + c % 2 of faces 0 /
            // 1 (see hc_comb_sinkhorn_rows).
            constexpr uint32_t comb_base = 2u * num_streams;
            for (uint32_t j = 0; j < rows; ++j) {
                for (uint32_t r = 0; r < num_streams; ++r) {
                    for (uint32_t c = 0; c < num_streams; ++c) {
                        const uint32_t fr = r + 4 * (c >> 1);
                        const uint32_t fc = 2 * j + (c & 1);
                        prm[tile_face_index(fr, fc)] = fw_row(j, comb_base + r * num_streams + c);
                        prm[tile_face_index(fr, 16 + fc)] =
                            stg[comb_bias_flat ? r * num_streams + c : tile_face_index(r, c)];
                    }
                }
            }
        } else if constexpr (role == 0 && hidden_row_major) {
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
