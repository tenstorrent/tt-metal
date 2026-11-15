// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// decode_sample_candidates (see sample_candidates.hpp). Every worker keeps the top k of row 0 of its tiles
// [t0, t1) of the logits, as (key, column) pairs sorted by key, descending, ties by lower column; it sends them to
// worker 0's gather slot and raises its semaphore. Worker 0 merges the lists in worker order (so in column order),
// then writes the values (times inv_temp, then the fill) and the indices (plus index_offset, then 0).
//
// Compile-time args: k, workers, V, cb_rows, cb_list, cb_gather, cb_out, semaphore id, fill bf16 bits, list slot
// bytes, then TensorAccessorArgs of logits, inv_temp, index_offset, values, indices.
// Runtime args: logits, inv_temp, index_offset, values, indices addresses, worker, t0, t1, worker 0's NoC x, y.

namespace {

constexpr uint32_t kTileBytes = 32 * 32 * 2;
constexpr uint32_t kFaceBytes = 16 * 16 * 2;
constexpr uint32_t kRowRead = 64;

// bf16 bits as an unsigned key ordered like the values (NaNs aside); -0 is +0, so they tie.
uint32_t to_key(uint32_t b) {
    if (b == 0x8000u) {
        b = 0;
    }
    return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
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

template <uint32_t k>
struct TopK {
    uint32_t keys[k];
    uint32_t cols[k];
    uint32_t n = 0;

    // Whether a candidate after the ones already seen could enter (ties go to the earlier one).
    bool admits(uint32_t key) const { return n < k || key > keys[k - 1]; }

    void insert(uint32_t key, uint32_t col) {
        int32_t j = n < k ? static_cast<int32_t>(n++) - 1 : static_cast<int32_t>(k) - 2;
        while (j >= 0 && keys[j] < key) {
            keys[j + 1] = keys[j];
            cols[j + 1] = cols[j];
            --j;
        }
        keys[j + 1] = key;
        cols[j + 1] = col;
    }
};

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
    constexpr auto logits_args = TensorAccessorArgs<10>();
    constexpr auto temp_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto offset_args = TensorAccessorArgs<temp_args.next_compile_time_args_offset()>();
    constexpr auto vals_args = TensorAccessorArgs<offset_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<vals_args.next_compile_time_args_offset()>();
    constexpr uint32_t out_tiles = 2 * k / 32;

    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t temp_addr = get_arg_val<uint32_t>(1);
    const uint32_t offset_addr = get_arg_val<uint32_t>(2);
    const uint32_t vals_addr = get_arg_val<uint32_t>(3);
    const uint32_t idx_addr = get_arg_val<uint32_t>(4);
    const uint32_t w = get_arg_val<uint32_t>(5);
    const uint32_t t0 = get_arg_val<uint32_t>(6);
    const uint32_t t1 = get_arg_val<uint32_t>(7);
    const uint32_t hub_x = get_arg_val<uint32_t>(8);
    const uint32_t hub_y = get_arg_val<uint32_t>(9);

    // Row 0 of each tile: columns 0-15 lead face 0, 16-31 lead face 1.
    const uint32_t rows_l1 = get_write_ptr(cb_rows);
    const auto logits = TensorAccessor(logits_args, logits_addr, kTileBytes);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t dst = rows_l1 + (t - t0) * 2 * kRowRead;
        noc_async_read(logits.get_noc_addr(t, 0), dst, kRowRead);
        noc_async_read(logits.get_noc_addr(t, kFaceBytes), dst + kRowRead, kRowRead);
    }
    const uint32_t out_l1 = get_write_ptr(cb_out);
    const uint32_t idx_l1 = out_l1 + out_tiles * kTileBytes;
    const uint32_t scalars_l1 = idx_l1 + ((2 * k * 4 + 63) / 64) * 64;
    if (w == 0) {
        zero_l1(out_l1, out_tiles * kTileBytes);
        noc_async_read(TensorAccessor(temp_args, temp_addr, kTileBytes).get_noc_addr(0), scalars_l1, 64);
        noc_async_read(TensorAccessor(offset_args, offset_addr, 4).get_noc_addr(0), scalars_l1 + 64, 64);
    }
    noc_async_read_barrier();

    TopK<k> top;
    for (uint32_t t = t0; t < t1; ++t) {
        const auto* row = reinterpret_cast<const volatile uint16_t*>(rows_l1 + (t - t0) * 2 * kRowRead);
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t col = t * 32 + c;
            if (col >= v) {
                break;
            }
            // Face 0's row 0 holds columns 0-15; face 1's, read kRowRead bytes later, 16-31.
            const uint32_t key = to_key(row[c < 16 ? c : (kRowRead / 2) + (c - 16)]);
            if (top.admits(key)) {
                top.insert(key, col);
            }
        }
    }

    // This worker's list, padded with key 0 (below every value), into worker 0's slot w.
    auto* list = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_list));
    for (uint32_t j = 0; j < k; ++j) {
        list[2 * j] = j < top.n ? top.keys[j] : 0u;
        list[2 * j + 1] = j < top.n ? top.cols[j] : 0u;
    }
    const uint32_t gather_l1 = get_write_ptr(cb_gather);
    const uint32_t sem_addr = get_semaphore(sem_id);
    noc_async_write(get_write_ptr(cb_list), get_noc_addr(hub_x, hub_y, gather_l1 + w * slot_bytes), k * 8);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(hub_x, hub_y, sem_addr), 1);
    noc_async_atomic_barrier();
    if (w != 0) {
        return;
    }

    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, workers);
    noc_semaphore_set(sem, 0);
    invalidate_l1_cache();
    TopK<k> best;
    for (uint32_t s = 0; s < workers; ++s) {
        const auto* g = reinterpret_cast<const volatile uint32_t*>(gather_l1 + s * slot_bytes);
        for (uint32_t j = 0; j < k; ++j) {
            const uint32_t key = g[2 * j];
            // Each list is sorted: once one misses, the rest of it does too.
            if (key == 0 || !best.admits(key)) {
                break;
            }
            best.insert(key, g[2 * j + 1]);
        }
    }

    const uint32_t inv_temp = reinterpret_cast<const volatile uint16_t*>(scalars_l1)[0];
    const int32_t offset = reinterpret_cast<const volatile int32_t*>(scalars_l1 + 64)[0];
    auto* vals = reinterpret_cast<volatile uint16_t*>(out_l1);
    auto* idx = reinterpret_cast<volatile int32_t*>(idx_l1);
    for (uint32_t c = 0; c < 2 * k; ++c) {
        // Row 0 of tile c / 32: columns 0-15 in face 0, 16-31 in face 1.
        const uint32_t pos = (c / 32) * (kTileBytes / 2) + ((c % 32) / 16) * (kFaceBytes / 2) + c % 16;
        if (c < best.n) {
            vals[pos] = static_cast<uint16_t>(bf16_mul(from_key(best.keys[c]), inv_temp));
            idx[c] = static_cast<int32_t>(best.cols[c]) + offset;
        } else {
            vals[pos] = static_cast<uint16_t>(fill_bits);
            idx[c] = 0;
        }
    }
    const auto vals_out = TensorAccessor(vals_args, vals_addr, kTileBytes);
    for (uint32_t t = 0; t < out_tiles; ++t) {
        noc_async_write(out_l1 + t * kTileBytes, vals_out.get_noc_addr(t), kTileBytes);
    }
    noc_async_write(idx_l1, TensorAccessor(idx_args, idx_addr, 2 * k * 4).get_noc_addr(0), 2 * k * 4);
    noc_async_write_barrier();
}
