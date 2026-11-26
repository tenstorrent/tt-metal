// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

FORCE_INLINE uint32_t pat(uint32_t a, uint32_t ctrl, uint32_t seed) {
    const uint32_t v = mix(mix((a >> 2) ^ (seed * 0x9E3779B9u)) ^ (ctrl * 0x85EBCA77u));
    return (seed & 1) ? ~v : v;
}

FORCE_INLINE uint32_t byte_mask(uint32_t x) {
    uint32_t m = 0;
    for (uint32_t b = 0; b < 4; ++b) {
        m |= (x & (0xFFu << (8 * b))) ? (0xFFu << (8 * b)) : 0;
    }
    return m;
}

// Entries are 64 B aligned per-bank addresses with the controller id in the low bits.
FORCE_INLINE uint64_t dram(uint32_t e) { return get_noc_addr_from_bank_id<true>(e & 7, e & ~63u); }

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t slot = get_arg_val<uint32_t>(1);
    const uint32_t mode = get_arg_val<uint32_t>(2);
    const uint32_t seed = get_arg_val<uint32_t>(3);
    const uint32_t iters = get_arg_val<uint32_t>(4);
    const uint32_t rb = get_arg_val<uint32_t>(5);
    const uint32_t n = get_arg_val<uint32_t>(6);
    const uint32_t compute_on = get_arg_val<uint32_t>(7);
    const uint32_t barrier_each = get_arg_val<uint32_t>(8);
    const uint32_t decode_max = get_arg_val<uint32_t>(9);
    const uint32_t inj_e = get_arg_val<uint32_t>(10);
    const uint32_t inj_bit = get_arg_val<uint32_t>(11);
    const uint32_t inj_mask = get_arg_val<uint32_t>(12);
    constexpr uint32_t scratch = get_compile_time_arg_val(0);
    constexpr bool feeds_compute = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t nw = rb / 4;
    const uint32_t hdr_addr = get_write_ptr(scratch);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr_addr);
    const uint32_t gold_addr = hdr_addr + (ROW_WORDS * 4);
    volatile tt_l1_ptr uint32_t* gold = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gold_addr);
    const uint32_t work_addr = gold_addr + n * rb;
    volatile tt_l1_ptr uint32_t* work = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(work_addr);
    const uint32_t raw_addr = work_addr + n * rb;
    volatile tt_l1_ptr uint32_t* raw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_addr);
    for (uint32_t i = 0; i < ROW_WORDS; ++i) {
        hdr[i] = 0;
    }
    hdr[7] = NO_ADDR;

    if constexpr (feeds_compute) {
        if (compute_on) {
            for (uint32_t c = 0; c < 2; ++c) {
                const uint32_t cb = c ? CB_IN1 : CB_IN0;
                cb_reserve_back(cb, MM_KT);
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
                for (uint32_t i = 0; i < MM_KT * 512; ++i) {
                    t[i] = mix(i * 2 + cb + seed * 0x10001u) & 0xBFFFBFFFu;
                }
                cb_push_back(cb, MM_KT);
            }
        }
    }

    for (uint32_t e = 0; e < n; ++e) {
        const uint32_t ent = get_arg_val<uint32_t>(ENTRY0 + e);
        for (uint32_t j = 0; j < nw; ++j) {
            gold[e * nw + j] = pat((ent & ~63u) + 4 * j, ent & 7, seed);
        }
    }

    uint32_t bad_reads = 0, bad_words = 0, stale = 0, other = 0, undecoded = 0, decoded = 0;
    uint32_t it = 0;

    auto check = [&](uint32_t e) {
        const uint32_t ent = get_arg_val<uint32_t>(ENTRY0 + e);
        const uint32_t a = ent & ~63u;
        const uint32_t ctrl = ent & 7;
        volatile tt_l1_ptr uint32_t* w = work + e * nw;
        volatile tt_l1_ptr uint32_t* g = gold + e * nw;
        uint32_t bad = 0;
        for (uint32_t j = 0; j < nw; ++j) {
            const uint32_t x = w[j] ^ g[j];
            if (x == 0) {
                continue;
            }
            for (uint32_t b = 0; b < 4; ++b) {
                hdr[LANE_OFF + 4 * ctrl + b] += (x >> (8 * b)) & 0xFFu ? 1 : 0;
            }
            if (hdr[7] == NO_ADDR) {
                hdr[7] = e;
                hdr[8] = j;
                hdr[9] = w[j];
                hdr[10] = g[j];
                hdr[11] = it;
            }
            ++bad;
        }
        if (bad == 0) {
            return;
        }
        ++bad_reads;
        bad_words += bad;
        hdr[HDR_WORDS + e] += 1;
        if (decoded >= decode_max) {
            ++undecoded;
            return;
        }
        ++decoded;
        bool is_stale = true;
        for (uint32_t j = 0; j < nw && is_stale; ++j) {
            const uint32_t x = w[j] ^ g[j];
            is_stale = !(x && ((w[j] ^ pat(a + 4 * j, ctrl, seed - 1)) & byte_mask(x)));
        }
        if (is_stale) {
            ++stale;
            return;
        }
        // Sources are read back rather than predicted: most of them lie outside the filled entries.
        for (uint32_t k = COL_SHIFT; k < 32; ++k) {
            noc_async_read(dram((a ^ (1u << k)) | ctrl), raw_addr, rb);
            noc_async_read_barrier();
            bool ok = true;
            for (uint32_t j = 0; j < nw && ok; ++j) {
                const uint32_t x = w[j] ^ g[j];
                ok = !(x && ((w[j] ^ raw[j]) & byte_mask(x)));
            }
            if (ok) {
                hdr[ALIAS_OFF + k] += 1;
                return;
            }
        }
        ++other;
    };

    if (mode == MODE_FILL) {
        for (uint32_t e = 0; e < n; ++e) {
            const uint32_t ent = get_arg_val<uint32_t>(ENTRY0 + e);
            if (e == inj_e) {
                noc_async_read(dram(((ent & ~63u) ^ (1u << inj_bit)) | (ent & 7)), raw_addr, rb);
                noc_async_read_barrier();
                for (uint32_t j = 0; j < nw; ++j) {
                    gold[e * nw + j] = (gold[e * nw + j] & ~inj_mask) | (raw[j] & inj_mask);
                }
            }
            noc_async_write(gold_addr + e * rb, dram(ent), rb);
        }
        noc_async_write_barrier();
        it = 1;
    } else {
        while (it < iters || (compute_on && !cb_pages_available_at_front(CB_OUT, 1))) {
            if (mode == MODE_WRITE) {
                for (uint32_t e = 0; e < n; ++e) {
                    noc_async_write(gold_addr + e * rb, dram(get_arg_val<uint32_t>(ENTRY0 + e)), rb);
                    if (barrier_each) {
                        noc_async_write_barrier();
                    }
                }
                noc_async_write_barrier();
            } else if (barrier_each) {
                for (uint32_t e = 0; e < n; ++e) {
                    noc_async_read(dram(get_arg_val<uint32_t>(ENTRY0 + e)), work_addr + e * rb, rb);
                    noc_async_read_barrier();
                    check(e);
                }
            } else {
                for (uint32_t e = 0; e < n; ++e) {
                    noc_async_read(dram(get_arg_val<uint32_t>(ENTRY0 + e)), work_addr + e * rb, rb);
                }
                noc_async_read_barrier();
                for (uint32_t e = 0; e < n; ++e) {
                    check(e);
                }
            }
            ++it;
        }
    }
    hdr[0] = seed;
    hdr[1] = it;
    hdr[2] = bad_reads;
    hdr[3] = bad_words;
    hdr[4] = stale;
    hdr[5] = other;
    hdr[6] = undecoded;
    hdr[15] = OUT_MAGIC;
    noc_async_write(hdr_addr, out.get_noc_addr(slot), (ROW_WORDS * 4));
    noc_async_write_barrier();
}
