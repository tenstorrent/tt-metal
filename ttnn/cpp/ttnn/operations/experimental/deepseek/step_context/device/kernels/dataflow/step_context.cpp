// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// decode_step_context (see step_context.hpp). Role 0 (one core) evaluates the integer terms and writes each
// integer output; role 1 (one core per RoPE output) builds the (cos, sin, -sin) rows of one position.
//
// Compile-time args: role, cb_pkt, cb_scratch, cb_rows, cb_tiles, pkt_bytes, then TensorAccessorArgs of the
// packet, an integer output, a RoPE table and a RoPE output (all DRAM interleaved).
//
// Role 0 runtime args: pkt_addr, n_out, then per output: addr, n_elems, 7 term words per element.
// Role 1 runtime args: pkt_addr, table_addr, hi_rows, rd, field, offset, cos_addr, sin_addr, neg_sin_addr.

namespace {

constexpr uint32_t kTermWords = 7;
constexpr uint32_t kSlotBytes = 64;
constexpr uint32_t kBf16TileBytes = 32 * 32 * 2;
constexpr uint32_t kLoRows = 1025;
constexpr uint32_t kQ = 28;

int32_t floor_div(int32_t a, int32_t b) {
    int32_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
    }
    return q;
}

int32_t field_value(const volatile int32_t* pkt, int32_t field) { return field < 0 ? 0 : pkt[field]; }

// A Q28 fixed-point value as bfloat16, rounded to nearest even.
uint16_t q28_to_bf16(int32_t v) {
    if (v == 0) {
        return 0;
    }
    const uint32_t sign = v < 0 ? 0x8000u : 0u;
    const uint32_t m = v < 0 ? static_cast<uint32_t>(-static_cast<int64_t>(v)) : static_cast<uint32_t>(v);
    const int lz = __builtin_clz(m);
    const uint32_t mn = m << lz;  // leading one at bit 31: value = 1.f * 2^(31 - lz - 28)
    uint32_t biased = static_cast<uint32_t>(127 + 3 - lz);
    uint32_t mant = (mn >> 24) & 0x7Fu;
    const uint32_t rest = mn & 0xFFFFFFu;
    if (rest > 0x800000u || (rest == 0x800000u && (mant & 1u))) {
        if (++mant == 0x80u) {
            mant = 0;
            ++biased;
        }
    }
    return static_cast<uint16_t>(sign | (biased << 7) | mant);
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
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pkt = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(3);
    constexpr uint32_t cb_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t pkt_bytes = get_compile_time_arg_val(5);
    constexpr auto pkt_args = TensorAccessorArgs<6>();
    constexpr auto int_args = TensorAccessorArgs<pkt_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<int_args.next_compile_time_args_offset()>();
    constexpr auto rope_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();

    const uint32_t pkt_addr = get_arg_val<uint32_t>(0);
    const uint32_t pkt_l1 = get_write_ptr(cb_pkt);
    noc_async_read(TensorAccessor(pkt_args, pkt_addr, pkt_bytes).get_noc_addr(0), pkt_l1, pkt_bytes);

    if constexpr (role == 0) {
        noc_async_read_barrier();
        const auto* pkt = reinterpret_cast<const volatile int32_t*>(pkt_l1);
        const uint32_t n_out = get_arg_val<uint32_t>(1);
        const uint32_t scratch = get_write_ptr(cb_scratch);
        uint32_t a = 2;
        for (uint32_t o = 0; o < n_out; ++o) {
            const uint32_t out_addr = get_arg_val<uint32_t>(a);
            const uint32_t n = get_arg_val<uint32_t>(a + 1);
            a += 2;
            const uint32_t slot = scratch + o * kSlotBytes;
            auto* dst = reinterpret_cast<volatile int32_t*>(slot);
            for (uint32_t e = 0; e < n; ++e, a += kTermWords) {
                const uint32_t w0 = get_arg_val<uint32_t>(a);
                const int32_t pre_add = static_cast<int32_t>(get_arg_val<uint32_t>(a + 1));
                const int32_t div = static_cast<int32_t>(get_arg_val<uint32_t>(a + 2));
                const int32_t mul = static_cast<int32_t>(get_arg_val<uint32_t>(a + 3));
                const int32_t post_add = static_cast<int32_t>(get_arg_val<uint32_t>(a + 4));
                const int32_t clamp_max = static_cast<int32_t>(get_arg_val<uint32_t>(a + 5));
                const int32_t add_mul = static_cast<int32_t>(get_arg_val<uint32_t>(a + 6));
                const uint32_t mode = (w0 >> 8) & 0xFu;
                int32_t v = field_value(pkt, static_cast<int32_t>(w0 & 0xFFu) - 1) + pre_add;
                if (mode == 1) {
                    v = floor_div(v, div);
                } else if (mode == 2) {
                    v = v - floor_div(v, div) * div;
                }
                v = v * mul + post_add;
                if (v > clamp_max) {
                    v = clamp_max;
                }
                v += field_value(pkt, static_cast<int32_t>((w0 >> 16) & 0xFFu) - 1) * add_mul;
                dst[e] = v;
            }
            noc_async_write(slot, TensorAccessor(int_args, out_addr, n * 4).get_noc_addr(0), n * 4);
        }
        noc_async_write_barrier();
    } else {
        const uint32_t table_addr = get_arg_val<uint32_t>(1);
        const uint32_t hi_rows = get_arg_val<uint32_t>(2);
        const uint32_t rd = get_arg_val<uint32_t>(3);
        const uint32_t field = get_arg_val<uint32_t>(4);
        const int32_t offset = static_cast<int32_t>(get_arg_val<uint32_t>(5));
        const uint32_t out_addr[3] = {get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7), get_arg_val<uint32_t>(8)};
        const uint32_t tiles = rd / 32;
        const uint32_t row_bytes = 2 * rd * 4;
        const uint32_t tiles_l1 = get_write_ptr(cb_tiles);
        zero_l1(tiles_l1, 3 * tiles * kBf16TileBytes);
        noc_async_read_barrier();

        const int32_t p = reinterpret_cast<const volatile int32_t*>(pkt_l1)[field] + offset;
        uint32_t hi = 0;
        int32_t lo = p;
        if (p >= 0) {
            hi = static_cast<uint32_t>(p) >> 10;
            lo = p & 1023;
        }
        if (hi >= hi_rows) {
            hi = hi_rows - 1;
        }
        const uint32_t rows_l1 = get_write_ptr(cb_rows);
        const auto table = TensorAccessor(table_args, table_addr, row_bytes);
        const uint32_t lo_l1 = rows_l1 + ((row_bytes + kSlotBytes - 1) / kSlotBytes) * kSlotBytes;
        noc_async_read(table.get_noc_addr(hi), rows_l1, row_bytes);
        noc_async_read(table.get_noc_addr(hi_rows + static_cast<uint32_t>(lo + 1)), lo_l1, row_bytes);
        noc_async_read_barrier();

        const auto* h = reinterpret_cast<const volatile int32_t*>(rows_l1);
        const auto* l = reinterpret_cast<const volatile int32_t*>(lo_l1);
        auto* out = reinterpret_cast<volatile uint16_t*>(tiles_l1);
        const uint32_t out_elems = tiles * kBf16TileBytes / 2;
        for (uint32_t c = 0; c < rd; ++c) {
            const int64_t ch = h[c], sh = h[rd + c], cl = l[c], sl = l[rd + c];
            const int32_t cos_q = static_cast<int32_t>((ch * cl - sh * sl) >> kQ);
            const int32_t sin_q = static_cast<int32_t>((sh * cl + ch * sl) >> kQ);
            // Row 0 of tile c / 32: columns 0-15 in face 0, 16-31 in face 1.
            const uint32_t cc = c % 32;
            const uint32_t idx = (c / 32) * 1024 + (cc / 16) * 256 + cc % 16;
            const uint16_t s = q28_to_bf16(sin_q);
            out[idx] = q28_to_bf16(cos_q);
            out[out_elems + idx] = s;
            out[2 * out_elems + idx] = s ? static_cast<uint16_t>(s ^ 0x8000u) : 0;
        }
        for (uint32_t k = 0; k < 3; ++k) {
            const auto dst = TensorAccessor(rope_args, out_addr[k], kBf16TileBytes);
            for (uint32_t t = 0; t < tiles; ++t) {
                noc_async_write(tiles_l1 + (k * tiles + t) * kBf16TileBytes, dst.get_noc_addr(t), kBf16TileBytes);
            }
        }
        noc_async_write_barrier();
    }
}
