// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block-mode reader (num_rows > 1): q's 8 heads x this row group's rows of this head group, the chunks this S-block
// holds (each shared chunk read once, kept in cb_k_shared for every row; per-row selected chunks streamed through
// cb_k_own in row order) and the masks in the order compute attends them, and on S-block 0 the per-row RoPE tables.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/flash_mqa_decode_rows_common.hpp"

namespace {

template <typename Reader>
FORCE_INLINE void read_tile_row_pages(
    const Reader& reader, uint32_t first_page, uint32_t num_pages, uint32_t l1, uint32_t page_bytes) {
    for (uint32_t t = 0; t < num_pages; ++t) {
        noc_async_read_page(first_page + t, reader, l1);
        l1 += page_bytes;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t DHt = get_named_compile_time_arg_val("DHt");
    constexpr uint32_t Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t");
    constexpr uint32_t k_chunk_size = get_named_compile_time_arg_val("k_chunk_size");
    constexpr uint32_t num_s_blocks = get_named_compile_time_arg_val("num_s_blocks");
    constexpr uint32_t heads_per_core = get_named_compile_time_arg_val("heads_per_core");
    constexpr uint32_t num_rows = get_named_compile_time_arg_val("num_rows");
    constexpr uint32_t row_groups = get_named_compile_time_arg_val("row_groups");
    constexpr uint32_t group_rows = num_rows / row_groups;
    constexpr bool dense = get_named_compile_time_arg_val("dense_selected") == 1;
    constexpr uint32_t q_row_bytes = get_named_compile_time_arg_val("q_row_bytes");
    constexpr uint32_t ring_capacity = get_named_compile_time_arg_val("ring_capacity");
    constexpr uint32_t max_ring_first_tile_row = get_named_compile_time_arg_val("max_ring_first_tile_row");
    constexpr uint32_t selected_capacity = get_named_compile_time_arg_val("selected_capacity");
    constexpr uint32_t selected_first_tile_row = get_named_compile_time_arg_val("selected_first_tile_row");
    constexpr uint32_t max_selected_first_tile_row = get_named_compile_time_arg_val("max_selected_first_tile_row");
    constexpr uint32_t rope_tiles = get_named_compile_time_arg_val("rope_tiles");
    constexpr uint32_t cs_face_bytes = get_named_compile_time_arg_val("cs_face_bytes");
    constexpr uint32_t cb_q_rm = get_named_compile_time_arg_val("cb_q_rm");
    constexpr uint32_t cb_k_shared = get_named_compile_time_arg_val("cb_k_shared");
    constexpr uint32_t cb_k_own = get_named_compile_time_arg_val("cb_k_own");
    constexpr uint32_t cb_mask = get_named_compile_time_arg_val("cb_mask");
    constexpr uint32_t cb_lengths = get_named_compile_time_arg_val("cb_lengths");
    constexpr uint32_t cb_lengths_scratch = get_named_compile_time_arg_val("cb_lengths_scratch");
    constexpr uint32_t cb_cs = get_named_compile_time_arg_val("cb_cs");
    constexpr uint32_t cb_cs_raw = get_named_compile_time_arg_val("cb_cs_raw");
    constexpr uint32_t q_semaphore_id = get_named_compile_time_arg_val("q_semaphore_id");
    constexpr uint32_t kv_semaphore_id = get_named_compile_time_arg_val("kv_semaphore_id");
    constexpr uint32_t num_head_groups = get_named_compile_time_arg_val("num_head_groups");
    // One row per core: every chunk stays in cb_k_shared, and the cores that need a chunk (this S-block of every
    // head group, and of every row group for a chunk shared by rows) each read a slice of it from DRAM and write
    // the slice to the others, so DRAM serves each chunk once.
    constexpr bool share_reads = get_named_compile_time_arg_val("share_reads") == 1;

    constexpr auto q_args = TensorAccessorArgs<0>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto selected_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto lengths_args = TensorAccessorArgs<selected_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<lengths_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto new_kv_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t kv_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t selected_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t lengths_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sin_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t new_kv_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t s_block = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t head_group = get_arg_val<uint32_t>(arg_idx++);
    // This core attends rows row_group, row_group + row_groups, ...
    const uint32_t row_group = get_arg_val<uint32_t>(arg_idx++);
    // NoC (x, y) of the S-blocks of this head group and row group, in S-block order; then of this S-block in every
    // (head group, row group), in order head_group * row_groups + row_group.
    const uint32_t block_noc_xy_idx = arg_idx;
    const uint32_t peer_noc_xy_idx = arg_idx + 2 * num_s_blocks;

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;

    // lengths: [p, ring_first_row, selected_first_row, 0, n_0 .. n_{R-1}].
    const auto lengths_reader = TensorAccessor(lengths_args, lengths_addr);
    const uint32_t lengths_l1 = get_write_ptr(cb_lengths_scratch);
    noc_async_read(lengths_reader.get_noc_addr(0), lengths_l1, (4 + num_rows) * sizeof(uint32_t));
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* lengths_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lengths_l1);
    uint32_t n[num_rows];
    for (uint32_t j = 0; j < num_rows; ++j) {
        n[j] = lengths_ptr[4 + j];
    }
    const uint32_t p = lengths_ptr[0];
    const flash_mqa_decode_rows::Plan plan = flash_mqa_decode_rows::make_plan(
        num_rows, ring_capacity, k_chunk_size, selected_capacity, dense, p, n, row_group, row_groups);
    const uint32_t ring_first_tile_row = flash_mqa_decode::min_u32(lengths_ptr[1] / 32, max_ring_first_tile_row);
    const uint32_t selected_base_tile_row =
        selected_first_tile_row + flash_mqa_decode::min_u32(lengths_ptr[2] / 32, max_selected_first_tile_row);
    cb_reserve_back(cb_lengths, 1);
    volatile tt_l1_ptr uint32_t* compute_lengths =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_lengths));
    compute_lengths[0] = p;
    for (uint32_t j = 0; j < num_rows; ++j) {
        compute_lengths[1 + j] = plan.n[j];
    }
    cb_push_back(cb_lengths, 1);

    uint32_t max_len = 0;
    for (uint32_t j = row_group; j < num_rows; j += row_groups) {
        const uint32_t len = flash_mqa_decode_rows::row_len(plan, j);
        max_len = len > max_len ? len : max_len;
    }
    if (s_block >= max_len) {
        return;
    }

    // S-block 0: per row j, rope_tiles (8, 32) tiles whose words are (cos, sin) of the row's even columns, one copy
    // per head (the format rope_sfpu_inplace_fused_rows reads). The first two faces of each cos / sin tile are read
    // whole (DRAM reads keep their 64 B alignment); row j's 16 columns of face f sit at f * face + 32 j.
    constexpr uint32_t faces_bytes = 2 * cs_face_bytes;
    const uint32_t raw_addr = get_write_ptr(cb_cs_raw);
    auto issue_rope_reads = [&]() {
        const auto cos_reader = TensorAccessor(cos_args, cos_addr);
        const auto sin_reader = TensorAccessor(sin_args, sin_addr);
        for (uint32_t t = 0; t < rope_tiles; ++t) {
            noc_async_read(cos_reader.get_noc_addr(t), raw_addr + t * faces_bytes, faces_bytes);
            noc_async_read(sin_reader.get_noc_addr(t), raw_addr + (rope_tiles + t) * faces_bytes, faces_bytes);
        }
    };
    auto build_rope_tables = [&]() {
        const volatile tt_l1_ptr uint32_t* raw_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(raw_addr);
        for (uint32_t j = row_group; j < num_rows; j += row_groups) {
            cb_reserve_back(cb_cs, rope_tiles);
            volatile tt_l1_ptr uint32_t* cs_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cs));
            for (uint32_t t = 0; t < rope_tiles; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t row_off = (f * cs_face_bytes + j * 32) / sizeof(uint32_t);
                    const uint32_t cos_off = t * faces_bytes / sizeof(uint32_t) + row_off;
                    const uint32_t sin_off = (rope_tiles + t) * faces_bytes / sizeof(uint32_t) + row_off;
                    const uint32_t dst_face_off = (t * 2 + f) * (heads_per_core * 8);
                    for (uint32_t k = 0; k < 8; ++k) {
                        const uint32_t w = (raw_ptr[cos_off + k] & 0xFFFFu) | ((raw_ptr[sin_off + k] & 0xFFFFu) << 16);
                        for (uint32_t r = 0; r < heads_per_core; ++r) {
                            cs_ptr[dst_face_off + r * 8 + k] = w;
                        }
                    }
                }
            }
            cb_push_back(cb_cs, rope_tiles);
        }
    };

    // q and (S-block 0) the RoPE tables; S-block 0 reads q and hands it to the group's other active S-blocks, so q's
    // core serves one read per group. One row per core (share_reads): issued before the K chunks and handed over
    // after the first one. Otherwise read before the K chunks (compute drains the own-chunk FIFO only after q).
    auto read_q_and_tables = [&]() {
        // q: block i holds the 8 heads of the group's row i (rows (head_group * 8 + h) * R + j). S-block 0 reads it and
        // writes it to the group's other active S-blocks, so q's cores serve one read per group, not one per core.
        {
            const uint32_t active_blocks = max_len < num_s_blocks ? max_len : num_s_blocks;
            constexpr uint32_t q_bytes = group_rows * heads_per_core * q_row_bytes;
            cb_reserve_back(cb_q_rm, group_rows * DHt);
            const uint32_t q_base = get_write_ptr(cb_q_rm);
            const uint32_t q_sem_addr = get_semaphore(q_semaphore_id);
            if (s_block == 0) {
                const auto q_reader = TensorAccessor(q_args, q_addr);
                uint32_t q_l1 = q_base;
                for (uint32_t j = row_group; j < num_rows; j += row_groups) {
                    for (uint32_t h = 0; h < heads_per_core; ++h) {
                        noc_async_read(
                            q_reader.get_noc_addr((head_group * heads_per_core + h) * num_rows + j), q_l1, q_row_bytes);
                        q_l1 += q_row_bytes;
                    }
                }
                noc_async_read_barrier();
                for (uint32_t b = 1; b < active_blocks; ++b) {
                    const uint32_t x = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * b);
                    const uint32_t y = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * b + 1);
                    noc_async_write(q_base, get_noc_addr(x, y, q_base), q_bytes);
                }
                noc_async_write_barrier();
                for (uint32_t b = 1; b < active_blocks; ++b) {
                    const uint32_t x = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * b);
                    const uint32_t y = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * b + 1);
                    noc_semaphore_inc(get_noc_addr(x, y, q_sem_addr), 1);
                }
                noc_async_atomic_barrier();
            } else {
                volatile tt_l1_ptr uint32_t* q_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(q_sem_addr);
                noc_semaphore_wait(q_sem, 1);
                noc_semaphore_set(q_sem, 0);
            }
            cb_push_back(cb_q_rm, group_rows * DHt);
        }

        if (s_block == 0) {
            issue_rope_reads();
            noc_async_read_barrier();
            build_rope_tables();
        }
    };
    if constexpr (!share_reads) {
        read_q_and_tables();
    } else {
        // Issued now and waited for with the first K chunk: S-block 0 reads the group's q rows (handed to the group
        // after the first chunk) and the tables.
        cb_reserve_back(cb_q_rm, group_rows * DHt);
        if (s_block == 0) {
            const auto q_reader = TensorAccessor(q_args, q_addr);
            uint32_t q_l1 = get_write_ptr(cb_q_rm);
            for (uint32_t j = row_group; j < num_rows; j += row_groups) {
                for (uint32_t h = 0; h < heads_per_core; ++h) {
                    noc_async_read(
                        q_reader.get_noc_addr((head_group * heads_per_core + h) * num_rows + j), q_l1, q_row_bytes);
                    q_l1 += q_row_bytes;
                }
            }
            issue_rope_reads();
        }
    }

    const auto kv_reader = TensorAccessor(kv_args, kv_addr);
    const auto selected_reader = TensorAccessor(selected_args, selected_addr);
    const auto new_kv_reader = TensorAccessor(new_kv_args, new_kv_addr);
    const uint32_t k_tile_bytes = get_tile_size(cb_k_shared);

    // Reads tiles [t0, t1) of chunk g (row j's own chunk when g is past the shared ones) to the chunk at l1.
    auto read_chunk = [&](uint32_t g, uint32_t j, uint32_t l1, uint32_t t0, uint32_t t1) {
        uint32_t first_page;
        if (g < plan.ring_chunks) {
            first_page = (ring_first_tile_row + g * Sk_chunk_t) * DHt;
        } else if (g == plan.ring_chunks) {
            // The new rows' tile row, then copies of it for the chunk's other tile rows (finite, masked off).
            for (uint32_t t = t0; t < t1; ++t) {
                noc_async_read_page(t % DHt, new_kv_reader, l1 + t * k_tile_bytes);
            }
            return;
        } else {
            const uint32_t i = g - plan.ring_chunks - 1;
            const uint32_t row_block = dense ? 0 : j * (selected_capacity / 32);
            first_page = (selected_base_tile_row + row_block + i * Sk_chunk_t) * DHt;
        }
        if (g < plan.ring_chunks) {
            read_tile_row_pages(kv_reader, first_page + t0, t1 - t0, l1 + t0 * k_tile_bytes, k_tile_bytes);
        } else {
            read_tile_row_pages(selected_reader, first_page + t0, t1 - t0, l1 + t0 * k_tile_bytes, k_tile_bytes);
        }
    };

    if constexpr (share_reads) {
        const uint32_t j = row_group;
        const uint32_t len = flash_mqa_decode_rows::row_len(plan, j);
        uint32_t lens[num_rows];
        for (uint32_t r = 0; r < num_rows; ++r) {
            lens[r] = flash_mqa_decode_rows::row_len(plan, r);
        }
        const uint32_t kv_sem_addr = get_semaphore(kv_semaphore_id);
        volatile tt_l1_ptr uint32_t* kv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kv_sem_addr);
        constexpr uint32_t num_peers_max = num_head_groups * row_groups;
        const uint32_t me = head_group * row_groups + row_group;
        uint32_t expected = 0;
        const uint32_t active_blocks = max_len < num_s_blocks ? max_len : num_s_blocks;
        constexpr uint32_t q_bytes = group_rows * heads_per_core * q_row_bytes;
        const uint32_t q_sem_addr = get_semaphore(q_semaphore_id);
        auto signal_q = [&]() {
            for (uint32_t b = 1; b < active_blocks; ++b) {
                const uint32_t x = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * b);
                const uint32_t y = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * b + 1);
                noc_semaphore_inc(get_noc_addr(x, y, q_sem_addr), 1);
            }
            noc_async_atomic_barrier();
        };
        for (uint32_t g = s_block; g < len; g += num_s_blocks) {
            // Row r needs chunk g if g < len(r) and the chunk is shared by rows or r is this row.
            const bool shared_index = g <= plan.ring_chunks || dense;
            uint32_t peers = 0;
            uint32_t my_index = 0;
            for (uint32_t c = 0; c < num_peers_max; ++c) {
                const uint32_t r = c % row_groups;
                if (g < lens[r] && (shared_index || r == row_group)) {
                    if (c == me) {
                        my_index = peers;
                    }
                    ++peers;
                }
            }
            cb_reserve_back(cb_k_shared, k_chunk_tiles);
            const uint32_t l1 = get_write_ptr(cb_k_shared);
            const uint32_t t0 = my_index * k_chunk_tiles / peers;
            const uint32_t t1 = (my_index + 1) * k_chunk_tiles / peers;
            read_chunk(g, j, l1, t0, t1);
            noc_async_read_barrier();
            if (g == s_block && s_block == 0) {
                // q is in: hand it to the group's other active S-blocks (counted after the write barrier below).
                const uint32_t q_base = get_write_ptr(cb_q_rm);
                for (uint32_t b = 1; b < active_blocks; ++b) {
                    const uint32_t x = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * b);
                    const uint32_t y = get_arg_val<uint32_t>(block_noc_xy_idx + 2 * b + 1);
                    noc_async_write(q_base, get_noc_addr(x, y, q_base), q_bytes);
                }
                if (peers == 1) {
                    noc_async_write_barrier();
                }
            }
            if (peers > 1) {
                if (shared_index) {
                    // The peers are (a subset of) this S-block's column of cores: one multicast of the slice to the
                    // column (cores that do not need the chunk get it in an unused slot).
                    uint32_t x0 = get_arg_val<uint32_t>(peer_noc_xy_idx);
                    uint32_t y0 = get_arg_val<uint32_t>(peer_noc_xy_idx + 1);
                    uint32_t x1 = get_arg_val<uint32_t>(peer_noc_xy_idx + 2 * (num_peers_max - 1));
                    uint32_t y1 = get_arg_val<uint32_t>(peer_noc_xy_idx + 2 * (num_peers_max - 1) + 1);
                    if (noc_index == 1) {
                        const uint32_t tx = x0, ty = y0;
                        x0 = x1;
                        y0 = y1;
                        x1 = tx;
                        y1 = ty;
                    }
                    noc_async_write_multicast_loopback_src(
                        l1 + t0 * k_tile_bytes,
                        get_noc_multicast_addr(x0, y0, x1, y1, l1 + t0 * k_tile_bytes),
                        (t1 - t0) * k_tile_bytes,
                        num_peers_max);
                } else {
                    for (uint32_t c = 0; c < num_peers_max; ++c) {
                        const uint32_t r = c % row_groups;
                        if (c != me && g < lens[r] && r == row_group) {
                            const uint32_t x = get_arg_val<uint32_t>(peer_noc_xy_idx + 2 * c);
                            const uint32_t y = get_arg_val<uint32_t>(peer_noc_xy_idx + 2 * c + 1);
                            noc_async_write(
                                l1 + t0 * k_tile_bytes,
                                get_noc_addr(x, y, l1 + t0 * k_tile_bytes),
                                (t1 - t0) * k_tile_bytes);
                        }
                    }
                }
                noc_async_write_barrier();
                if (g == s_block && s_block == 0) {
                    signal_q();
                }
                for (uint32_t c = 0; c < num_peers_max; ++c) {
                    const uint32_t r = c % row_groups;
                    if (c != me && g < lens[r] && (shared_index || r == row_group)) {
                        const uint32_t x = get_arg_val<uint32_t>(peer_noc_xy_idx + 2 * c);
                        const uint32_t y = get_arg_val<uint32_t>(peer_noc_xy_idx + 2 * c + 1);
                        noc_semaphore_inc(get_noc_addr(x, y, kv_sem_addr), 1);
                    }
                }
                noc_async_atomic_barrier();
                expected += peers - 1;
                noc_semaphore_wait_min(kv_sem, expected);
            }
            if (g == s_block) {
                if (s_block == 0) {
                    if (peers == 1) {
                        signal_q();
                    }
                    build_rope_tables();
                } else {
                    volatile tt_l1_ptr uint32_t* q_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(q_sem_addr);
                    noc_semaphore_wait(q_sem, 1);
                    noc_semaphore_set(q_sem, 0);
                }
                cb_push_back(cb_q_rm, group_rows * DHt);
            }
            cb_push_back(cb_k_shared, k_chunk_tiles);
        }
        noc_semaphore_set(kv_sem, 0);
    }
    // Shared chunks: every index of this S-block below shared_len, once.
    for (uint32_t g = s_block; !share_reads && g < plan.shared_len; g += num_s_blocks) {
        cb_reserve_back(cb_k_shared, k_chunk_tiles);
        read_chunk(g, 0, get_write_ptr(cb_k_shared), 0, k_chunk_tiles);
        noc_async_read_barrier();
        cb_push_back(cb_k_shared, k_chunk_tiles);
    }

    // Per row, in compute's order: each chunk's mask (when partial), and the row's own chunks.
    for (uint32_t j = row_group; j < num_rows; j += row_groups) {
        const uint32_t len = flash_mqa_decode_rows::row_len(plan, j);
        const int32_t ring_a = flash_mqa_decode_rows::ring_interval_start(plan, j);
        for (uint32_t g = s_block; g < len; g += num_s_blocks) {
            const uint32_t valid = flash_mqa_decode_rows::valid_rows(plan, j, g);
            if (valid < k_chunk_size) {
                // One bf16 per key: 0 for an attended key, -inf otherwise.
                cb_reserve_back(cb_mask, 1);
                volatile tt_l1_ptr uint16_t* mask_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_mask));
                if (g < plan.ring_chunks) {
                    for (uint32_t i = 0; i < k_chunk_size; ++i) {
                        mask_ptr[i] =
                            flash_mqa_decode_rows::ring_key_valid(plan, ring_a, g * k_chunk_size + i) ? 0 : 0xFF80;
                    }
                } else {
                    // The staging and selected chunks attend a prefix of their keys.
                    for (uint32_t i = 0; i < k_chunk_size; ++i) {
                        mask_ptr[i] = i < valid ? 0 : 0xFF80;
                    }
                }
                cb_push_back(cb_mask, 1);
            }
            if (!share_reads && g >= plan.shared_len) {
                cb_reserve_back(cb_k_own, k_chunk_tiles);
                read_chunk(g, j, get_write_ptr(cb_k_own), 0, k_chunk_tiles);
                noc_async_read_barrier();
                cb_push_back(cb_k_own, k_chunk_tiles);
            }
        }
    }
}
