// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// decode_sample_candidates (see sample_candidates.hpp) for rows 0 .. rows - 1 of the logits. On each worker core one
// RISC handles rows [r0, r1) (compile-time): for every row it keeps the top k of its tiles [t0, t1) as
// packed (key, column) values sorted descending (ties by lower column), and sends row r's list to row r's hub (worker
// core r) slot w, raising the hub's list semaphore. Row r's hub k-way merges its row's lists (ties to the earlier
// worker, so the lower column), writes the indices row (plus index_offset, then 0) and puts the values row (times
// inv_temp, then the fill) into worker 0's value tiles; worker 0 zeroes their padding rows, waits for the other rows
// (done semaphore) and writes them.
//
// Compile-time args: k, workers, V, cb_rows, cb_list, cb_gather, cb_out, list semaphore id, fill bf16 bits, list
// slot bytes, r0, r1, rows, done semaphore id, rows-CB byte offset of this RISC, logits tiles, scanning workers, then
// TensorAccessorArgs of logits, inv_temp, index_offset, values, indices. Runtime args: logits, inv_temp, index_offset,
// values, indices addresses, worker, t0, t1, worker 0's NoC x, y, then this core's row hub NoC x, y per row r0 .. r1
// - 1.

namespace {

constexpr uint32_t kTileBytes = 32 * 32 * 2;
constexpr uint32_t kFaceBytes = 16 * 16 * 2;
constexpr uint32_t kRowRead = 64;

// bf16 bits as an unsigned key ordered like the values (NaNs aside); -0 is +0, so they tie.
inline uint32_t to_key(uint32_t b) {
    if (b == 0x8000u) {
        b = 0;
    }
    // Negative: flip every bit; positive: set the sign bit.
    const uint32_t flip = (0u - (b >> 15)) | 0x8000u;
    return (b ^ flip) & 0xFFFFu;
}
uint32_t from_key(uint32_t k) { return (k & 0x8000u) ? (k & 0x7FFFu) : (~k & 0xFFFFu); }

// a * b for bf16 bits, rounded to nearest even; subnormals flush to zero.
uint32_t bf16_mul(uint32_t a, uint32_t b) {
    const uint32_t sign = (a ^ b) & 0x8000u;
    const int32_t ea = (a >> 7) & 0xFF;
    const int32_t eb = (b >> 7) & 0xFF;
    if (ea == 0 || eb == 0) {
        return sign;
    }
    if (ea == 0xFF || eb == 0xFF) {
        return sign | 0x7F80u | (((a | b) & 0x7Fu) ? 0x40u : 0u);
    }
    const uint32_t p = ((a & 0x7Fu) | 0x80u) * ((b & 0x7Fu) | 0x80u);
    int32_t e = ea + eb - 127;
    uint32_t shift = 7;
    if (p & 0x8000u) {
        shift = 8;
        ++e;
    }
    uint32_t mant = p >> shift;
    const uint32_t rem = p & ((1u << shift) - 1);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (mant & 1u))) {
        if (++mant == 0x100u) {
            mant >>= 1;
            ++e;
        }
    }
    if (e >= 0xFF) {
        return sign | 0x7F80u;
    }
    if (e <= 0) {
        return sign;
    }
    return sign | (static_cast<uint32_t>(e) << 7) | (mant & 0x7Fu);
}

// A candidate packed as key << 16 | (0xFFFF - column code): larger is better, and of equal keys the lower column
// wins. The code is the column itself when V <= 65536 (so packed values compare across lists), else its offset
// from the worker's first column. 0 (key 0, below every value) marks the end of a list.
inline uint32_t pack(uint32_t key, uint32_t offset) { return (key << 16) | (0xFFFFu - offset); }

// The k largest packed values seen, as a min-heap (smallest at 0), starting as k zeros (below every candidate).
template <uint32_t k>
struct TopK {
    uint32_t v[k];

    TopK() {
        for (uint32_t i = 0; i < k; ++i) {
            v[i] = 0;
        }
    }

    void sift_down(uint32_t x, uint32_t m) {
        uint32_t i = 0;
        while (true) {
            uint32_t c = 2 * i + 1;
            if (c >= m) {
                break;
            }
            if (c + 1 < m && v[c + 1] < v[c]) {
                ++c;
            }
            if (v[c] >= x) {
                break;
            }
            v[i] = v[c];
            i = c;
        }
        v[i] = x;
    }

    // Sorts descending (each step moves the minimum to the end); the zeros left over end the list.
    void sort_desc() {
        for (uint32_t m = k; m > 1; --m) {
            const uint32_t smallest = v[0];
            sift_down(v[m - 1], m - 1);
            v[m - 1] = smallest;
        }
    }
};

// Whether list a's head beats list b's: the larger key, of equal keys the earlier list (the lower columns).
template <bool global_cols>
inline bool beats(uint32_t a, uint32_t la, uint32_t b, uint32_t lb) {
    if constexpr (global_cols) {
        return a > b;
    }
    return (a >> 16) != (b >> 16) ? (a >> 16) > (b >> 16) : la < lb;
}

void zero_l1(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        const uint32_t len = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros, addr, len);
        addr += len;
        bytes -= len;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t workers = get_compile_time_arg_val(1);
    constexpr uint32_t v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(3);
    constexpr uint32_t cb_list = get_compile_time_arg_val(4);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t fill_bits = get_compile_time_arg_val(8);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t r0 = get_compile_time_arg_val(10);
    constexpr uint32_t r1 = get_compile_time_arg_val(11);
    constexpr uint32_t rows = get_compile_time_arg_val(12);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(13);
    constexpr uint32_t rows_cb_offset = get_compile_time_arg_val(14);
    constexpr uint32_t tiles = get_compile_time_arg_val(15);
    constexpr uint32_t scanners = get_compile_time_arg_val(16);
    constexpr auto logits_args = TensorAccessorArgs<17>();
    constexpr auto temp_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto offset_args = TensorAccessorArgs<temp_args.next_compile_time_args_offset()>();
    constexpr auto vals_args = TensorAccessorArgs<offset_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<vals_args.next_compile_time_args_offset()>();
    constexpr uint32_t out_tiles = 2 * k / 32;
    constexpr uint32_t nr = r1 - r0;
    // Each face's rows r0 .. r1 - 1 (32 B each), in whole 64 B reads from the even row at or below r0.
    constexpr uint32_t row_base = r0 & ~1u;
    constexpr uint32_t read_bytes = (((r1 - row_base) * 32 + kRowRead - 1) / kRowRead) * kRowRead;
    constexpr uint32_t face_row_off = row_base * 32;
    // Columns fit the packed word's 16 bits: pack them whole, so packed values order across lists too.
    constexpr bool global_cols = v <= 0x10000u;

    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t temp_addr = get_arg_val<uint32_t>(1);
    const uint32_t offset_addr = get_arg_val<uint32_t>(2);
    const uint32_t vals_addr = get_arg_val<uint32_t>(3);
    const uint32_t idx_addr = get_arg_val<uint32_t>(4);
    const uint32_t w = get_arg_val<uint32_t>(5);
    const uint32_t t0 = get_arg_val<uint32_t>(6);
    const uint32_t t1 = get_arg_val<uint32_t>(7);
    const uint32_t out_x = get_arg_val<uint32_t>(8);
    const uint32_t out_y = get_arg_val<uint32_t>(9);

    // Rows r0 .. r1 - 1 of each tile: columns 0-15 in face 0, 16-31 in face 1.
    const uint32_t rows_l1 = get_write_ptr(cb_rows) + rows_cb_offset;
    const auto logits = TensorAccessor(logits_args, logits_addr, kTileBytes);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t dst = rows_l1 + (t - t0) * 2 * read_bytes;
        noc_async_read(logits.get_noc_addr(t, face_row_off), dst, read_bytes);
        noc_async_read(logits.get_noc_addr(t, kFaceBytes + face_row_off), dst + read_bytes, read_bytes);
    }
    // This core is row hub_row's hub when it handles that row.
    const bool hub = w >= r0 && w < r1;
    const uint32_t hub_row = w;
    const uint32_t out_l1 = get_write_ptr(cb_out);
    const uint32_t idx_l1 = out_l1 + out_tiles * kTileBytes;
    const uint32_t scalars_l1 = idx_l1 + ((2 * k * 4 + 63) / 64) * 64;
    if (hub) {
        if (hub_row == 0) {
            // Zero the value tiles' rows from `rows` on: rows 1 .. rows - 1 arrive from their hubs.
            for (uint32_t t = 0; t < out_tiles; ++t) {
                for (uint32_t f = 0; f < 4; ++f) {
                    const uint32_t first = f < 2 ? rows : 0;
                    zero_l1(out_l1 + t * kTileBytes + f * kFaceBytes + first * 32, (16 - first) * 32);
                }
            }
        }
        noc_async_read(TensorAccessor(temp_args, temp_addr, kTileBytes).get_noc_addr(0), scalars_l1, 64);
        noc_async_read(TensorAccessor(offset_args, offset_addr, 4).get_noc_addr(0), scalars_l1 + 64, 64);
    }
    noc_async_read_barrier();
    // The rows arrived by DMA: drop any cached copy of the buffer before reading it (plain loads from here on).
    invalidate_l1_cache();
    asm volatile("" ::: "memory");

    TopK<k> top[nr];
    // A candidate enters only above the heap's smallest key: an equal key comes from a later column, so it loses.
    for (uint32_t i = 0; i < nr; ++i) {
        auto& top_i = top[i];
        uint32_t thr = 0;
        // Row r0 + i of face 0 holds columns 0-15; of face 1, read_bytes later, 16-31.
        const uint32_t at = (r0 - row_base + i) * 32;
        for (uint32_t t = t0; t < t1; ++t) {
            const uint32_t base = rows_l1 + (t - t0) * 2 * read_bytes + at;
            const uint32_t cols = v - t * 32 < 32 ? v - t * 32 : 32;
            const uint32_t code0 = (global_cols ? t : t - t0) * 32;
            if (cols == 32) {
                // Two columns per 32-bit load: columns 0-15 from face 0, 16-31 from face 1.
                for (uint32_t f = 0; f < 2; ++f) {
                    const auto* words = reinterpret_cast<const uint32_t*>(base + f * read_bytes);
                    const uint32_t code = code0 + f * 16;
                    for (uint32_t c = 0; c < 8; ++c) {
                        const uint32_t pair = words[c];
                        // Both lanes' keys at once (-0 reads 0x7FFF, not 0x8000: no heap key is 0x7FFF, so
                        // "<= thr" still holds exactly for it); skip the pair when neither lane enters.
                        const uint32_t keys = pair ^ ((((pair >> 15) & 0x00010001u) * 0xFFFFu) | 0x80008000u);
                        if ((keys & 0xFFFFu) <= thr && (keys >> 16) <= thr) {
                            continue;
                        }
                        const uint32_t lo = to_key(pair & 0xFFFFu);
                        if (lo > thr) {
                            top_i.sift_down(pack(lo, code + 2 * c), k);
                            thr = top_i.v[0] >> 16;
                        }
                        const uint32_t hi = to_key(pair >> 16);
                        if (hi > thr) {
                            top_i.sift_down(pack(hi, code + 2 * c + 1), k);
                            thr = top_i.v[0] >> 16;
                        }
                    }
                }
            } else {
                const auto* f0 = reinterpret_cast<const uint16_t*>(base);
                const auto* f1 = reinterpret_cast<const uint16_t*>(base + read_bytes);
                for (uint32_t c = 0; c < cols; ++c) {
                    const uint32_t key = to_key(c < 16 ? f0[c] : f1[c - 16]);
                    if (key > thr) {
                        top_i.sift_down(pack(key, code0 + c), k);
                        thr = top_i.v[0] >> 16;
                    }
                }
            }
        }
    }

    // Each row's list (packed, descending), padded with 0, into its hub's slot w.
    const uint32_t sem_addr = get_semaphore(sem_id);
    const uint32_t gather_l1 = get_write_ptr(cb_gather);
    for (uint32_t i = 0; i < nr; ++i) {
        top[i].sort_desc();
        const uint32_t list_l1 = get_write_ptr(cb_list) + (r0 + i) * slot_bytes;
        auto* list = reinterpret_cast<volatile uint32_t*>(list_l1);
        for (uint32_t j = 0; j < k; ++j) {
            list[j] = top[i].v[j];
        }
        const uint32_t hx = get_arg_val<uint32_t>(10 + 2 * i);
        const uint32_t hy = get_arg_val<uint32_t>(11 + 2 * i);
        noc_async_write(list_l1, get_noc_addr(hx, hy, gather_l1 + w * slot_bytes), k * 4);
    }
    noc_async_write_barrier();
    for (uint32_t i = 0; i < nr; ++i) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(10 + 2 * i), get_arg_val<uint32_t>(11 + 2 * i), sem_addr), 1);
    }
    noc_async_atomic_barrier();
    if (!hub) {
        return;
    }

    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, workers);
    noc_semaphore_set(sem, 0);
    invalidate_l1_cache();
    // A k-way merge of the sorted lists: a max-heap of their heads (ties to the earlier list, i.e. lower columns).
    uint32_t head[workers];
    uint8_t list_of[workers];
    uint8_t next[workers];
    uint32_t heads = 0;
    auto sift_down = [&](uint32_t x, uint32_t lx) {
        uint32_t i = 0;
        while (true) {
            uint32_t c = 2 * i + 1;
            if (c >= heads) {
                break;
            }
            if (c + 1 < heads && beats<global_cols>(head[c + 1], list_of[c + 1], head[c], list_of[c])) {
                ++c;
            }
            if (!beats<global_cols>(head[c], list_of[c], x, lx)) {
                break;
            }
            head[i] = head[c];
            list_of[i] = list_of[c];
            i = c;
        }
        head[i] = x;
        list_of[i] = lx;
    };
    for (uint32_t s = 0; s < workers; ++s) {
        const uint32_t x = reinterpret_cast<const volatile uint32_t*>(gather_l1 + s * slot_bytes)[0];
        next[s] = 1;
        if (x == 0) {
            continue;
        }
        uint32_t i = heads++;
        while (i > 0) {
            const uint32_t parent = (i - 1) / 2;
            if (!beats<global_cols>(x, s, head[parent], list_of[parent])) {
                break;
            }
            head[i] = head[parent];
            list_of[i] = list_of[parent];
            i = parent;
        }
        head[i] = x;
        list_of[i] = s;
    }
    uint32_t best_keys[k];
    uint32_t best_cols[k];
    uint32_t best_n = 0;
    while (best_n < k && heads > 0) {
        const uint32_t x = head[0];
        const uint32_t s = list_of[0];
        const uint32_t t_first = s < scanners ? s * tiles / scanners : tiles;
        best_keys[best_n] = x >> 16;
        best_cols[best_n] = (global_cols ? 0 : t_first * 32) + (0xFFFFu - (x & 0xFFFFu));
        ++best_n;
        const uint32_t y =
            next[s] < k ? reinterpret_cast<const volatile uint32_t*>(gather_l1 + s * slot_bytes)[next[s]++] : 0u;
        if (y != 0) {
            sift_down(y, s);
        } else {
            --heads;
            sift_down(head[heads], list_of[heads]);
        }
    }

    const uint32_t inv_temp = reinterpret_cast<const volatile uint16_t*>(scalars_l1)[0];
    const int32_t offset = reinterpret_cast<const volatile int32_t*>(scalars_l1 + 64)[0];
    auto* vals = reinterpret_cast<volatile uint16_t*>(out_l1);
    auto* idx = reinterpret_cast<volatile int32_t*>(idx_l1);
    for (uint32_t c = 0; c < 2 * k; ++c) {
        // Row hub_row of tile c / 32: columns 0-15 in face 0, 16-31 in face 1.
        const uint32_t pos = (c / 32) * (kTileBytes / 2) + ((c % 32) / 16) * (kFaceBytes / 2) + hub_row * 16 + c % 16;
        if (c < best_n) {
            vals[pos] = static_cast<uint16_t>(bf16_mul(from_key(best_keys[c]), inv_temp));
            idx[c] = static_cast<int32_t>(best_cols[c]) + offset;
        } else {
            vals[pos] = static_cast<uint16_t>(fill_bits);
            idx[c] = 0;
        }
    }
    noc_async_write(idx_l1, TensorAccessor(idx_args, idx_addr, 2 * k * 4).get_noc_addr(hub_row), 2 * k * 4);
    const uint32_t done_addr = get_semaphore(done_sem_id);
    if (hub_row != 0) {
        // This row's face-rows into the same places of worker 0's value tiles.
        for (uint32_t t = 0; t < out_tiles; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t off = t * kTileBytes + f * kFaceBytes + hub_row * 32;
                noc_async_write(out_l1 + off, get_noc_addr(out_x, out_y, out_l1 + off), 32);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(out_x, out_y, done_addr), 1);
        noc_async_atomic_barrier();
        return;
    }
    if (rows > 1) {
        auto* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_addr);
        noc_semaphore_wait(done, rows - 1);
        noc_semaphore_set(done, 0);
    }
    const auto vals_out = TensorAccessor(vals_args, vals_addr, kTileBytes);
    for (uint32_t t = 0; t < out_tiles; ++t) {
        noc_async_write(out_l1 + t * kTileBytes, vals_out.get_noc_addr(t), kTileBytes);
    }
    noc_async_write_barrier();
}
