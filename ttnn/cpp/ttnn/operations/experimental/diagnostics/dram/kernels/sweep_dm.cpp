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

// One hash per 64 B line keeps the fill cheap; the word index term keeps the 16 words of a line distinct.
FORCE_INLINE uint32_t line_key(uint32_t a, uint32_t bank, uint32_t seed) {
    return mix(mix((a >> 6) ^ (seed * 0x9E3779B9u)) ^ (bank * 0x85EBCA77u));
}

FORCE_INLINE uint32_t pat(uint32_t a, uint32_t bank, uint32_t seed) {
    const uint32_t v = line_key(a, bank, seed) ^ (((a >> 2) & 15) * 0x01234567u);
    return (seed & 1) ? ~v : v;
}

FORCE_INLINE void gen_page(volatile tt_l1_ptr uint32_t* dst, uint32_t a, uint32_t bank, uint32_t seed) {
    const uint32_t inv = (seed & 1) ? 0xFFFFFFFFu : 0;
    for (uint32_t l = 0; l < (PAGE_WORDS / 16); ++l) {
        const uint32_t key = line_key(a + 64 * l, bank, seed);
        for (uint32_t w = 0; w < 16; ++w) {
            dst[16 * l + w] = (key ^ (w * 0x01234567u)) ^ inv;
        }
    }
}

FORCE_INLINE uint32_t byte_mask(uint32_t x) {
    uint32_t m = 0;
    for (uint32_t b = 0; b < 4; ++b) {
        m |= (x & (0xFFu << (8 * b))) ? (0xFFu << (8 * b)) : 0;
    }
    return m;
}

// True when every wrong byte of the page equals the pattern of the page at a0 ^ flip written with seed s.
FORCE_INLINE bool explains(
    volatile tt_l1_ptr uint32_t* w,
    volatile tt_l1_ptr uint32_t* g,
    uint32_t a0,
    uint32_t bank,
    uint32_t s,
    uint32_t flip) {
    for (uint32_t j = 0; j < PAGE_WORDS; ++j) {
        const uint32_t x = w[j] ^ g[j];
        if (x && ((w[j] ^ pat((a0 + 4 * j) ^ flip, bank, s)) & byte_mask(x))) {
            return false;
        }
    }
    return true;
}

// Same test against data read back from DRAM, for alias sources outside the buffer where the pattern is unknown.
FORCE_INLINE bool explains_raw(
    volatile tt_l1_ptr uint32_t* w, volatile tt_l1_ptr uint32_t* g, volatile tt_l1_ptr uint32_t* ref) {
    for (uint32_t j = 0; j < PAGE_WORDS; ++j) {
        const uint32_t x = w[j] ^ g[j];
        if (x && ((w[j] ^ ref[j]) & byte_mask(x))) {
            return false;
        }
    }
    return true;
}

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t p0 = get_arg_val<uint32_t>(3);
    const uint32_t p1 = get_arg_val<uint32_t>(4);
    const uint32_t seed = get_arg_val<uint32_t>(5);
    const uint32_t passes = get_arg_val<uint32_t>(6);
    const uint32_t reps = get_arg_val<uint32_t>(7);
    const uint32_t compute_on = get_arg_val<uint32_t>(8);
    const uint32_t inj_page = get_arg_val<uint32_t>(9);
    const uint32_t inj_bit = get_arg_val<uint32_t>(10);
    const uint32_t inj_mask = get_arg_val<uint32_t>(11);
    const uint32_t decode_max = get_arg_val<uint32_t>(12);
    const uint32_t writeback = get_arg_val<uint32_t>(13);
    constexpr uint32_t scratch = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr bool fill = get_compile_time_arg_val(2);
    constexpr bool feeds_compute = get_compile_time_arg_val(3);
    constexpr auto buf_args = TensorAccessorArgs<4>();
    constexpr auto out_args = TensorAccessorArgs<buf_args.next_compile_time_args_offset()>();
    const auto buf = TensorAccessor(buf_args, base);
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t hdr_addr = get_write_ptr(scratch);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr_addr);
    const uint32_t gold_addr = hdr_addr + (ROW_WORDS * 4);
    volatile tt_l1_ptr uint32_t* gold = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gold_addr);
    const uint32_t work_addr = gold_addr + chunk * PAGE_BYTES;
    volatile tt_l1_ptr uint32_t* work = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(work_addr);
    const uint32_t raw_addr = work_addr + chunk * PAGE_BYTES;
    volatile tt_l1_ptr uint32_t* raw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_addr);
    for (uint32_t i = 0; i < ROW_WORDS; ++i) {
        hdr[i] = 0;
    }
    hdr[7] = NO_ADDR;

    if constexpr (feeds_compute) {
        if (compute_on) {
            // Random bf16 with exponent MSB cleared: |x| < 2, never inf/NaN, so the FPU toggles at full activity.
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

    if constexpr (fill) {
        for (uint32_t p = p0; p < p1;) {
            uint32_t n = 0;
            for (; p < p1 && n < chunk; ++p, ++n) {
                volatile tt_l1_ptr uint32_t* g = gold + n * PAGE_WORDS;
                const uint32_t a = base + (p / SWEEP_BANKS) * PAGE_BYTES;
                gen_page(g, a, p % SWEEP_BANKS, seed);
                if (p == inj_page) {
                    for (uint32_t j = 0; j < PAGE_WORDS; ++j) {
                        const uint32_t alias = pat((a + 4 * j) ^ (1u << inj_bit), p % SWEEP_BANKS, seed);
                        g[j] = (g[j] & ~inj_mask) | (alias & inj_mask);
                    }
                }
                noc_async_write(gold_addr + n * PAGE_BYTES, buf.get_noc_addr(p), PAGE_BYTES);
            }
            noc_async_write_barrier();
        }
        hdr[1] = 1;
    } else {
        uint32_t bad_pages = 0, bad_words = 0, stale = 0, other = 0, undecoded = 0, decoded = 0, outside = 0;
        uint32_t pass = 0;
        while (pass < passes || (compute_on && !cb_pages_available_at_front(CB_OUT, 1))) {
            for (uint32_t c0 = p0; c0 < p1; c0 += chunk) {
                const uint32_t n = (p1 - c0 < chunk) ? (p1 - c0) : chunk;
                for (uint32_t r = 0; r < reps; ++r) {
                    for (uint32_t i = 0; i < n; ++i) {
                        noc_async_read(buf.get_noc_addr(c0 + i), work_addr + i * PAGE_BYTES, PAGE_BYTES);
                    }
                    if (r == 0) {
                        for (uint32_t i = 0; i < n; ++i) {
                            const uint32_t p = c0 + i;
                            gen_page(gold + i * PAGE_WORDS, base + (p / SWEEP_BANKS) * PAGE_BYTES, p % SWEEP_BANKS, seed);
                        }
                    }
                    noc_async_read_barrier();
                    for (uint32_t i = 0; i < n; ++i) {
                        volatile tt_l1_ptr uint32_t* w = work + i * PAGE_WORDS;
                        volatile tt_l1_ptr uint32_t* g = gold + i * PAGE_WORDS;
                        const uint32_t p = c0 + i;
                        const uint32_t bank = p % SWEEP_BANKS;
                        const uint32_t a0 = base + (p / SWEEP_BANKS) * PAGE_BYTES;
                        uint32_t page_bad = 0;
                        for (uint32_t j = 0; j < PAGE_WORDS; ++j) {
                            const uint32_t x = w[j] ^ g[j];
                            if (x == 0) {
                                continue;
                            }
                            for (uint32_t b = 0; b < 4; ++b) {
                                hdr[LANE_OFF + 4 * bank + b] += (x >> (8 * b)) & 0xFFu ? 1 : 0;
                            }
                            if (hdr[7] == NO_ADDR) {
                                hdr[7] = p;
                                hdr[8] = j;
                                hdr[9] = w[j];
                                hdr[10] = g[j];
                                hdr[11] = pass;
                                hdr[12] = r;
                            }
                            ++page_bad;
                        }
                        if (page_bad == 0) {
                            continue;
                        }
                        bad_words += page_bad;
                        ++bad_pages;
                        if (decoded >= decode_max) {
                            ++undecoded;
                            continue;
                        }
                        ++decoded;
                        if (explains(w, g, a0, bank, seed - 1, 0)) {
                            ++stale;
                            continue;
                        }
                        uint32_t k = 2;
                        while (k < 32 && !explains(w, g, a0, bank, seed, 1u << k)) {
                            ++k;
                        }
                        // Only sources below the buffer: they are in the bank, while A | (1 << k) can run past its end.
                        for (uint32_t kk = 2; k == 32 && kk < 32; ++kk) {
                            const uint32_t src = a0 ^ (1u << kk);
                            if (src >= base) {
                                continue;
                            }
                            noc_async_read(get_noc_addr_from_bank_id<true>(bank, src), raw_addr, PAGE_BYTES);
                            noc_async_read_barrier();
                            if (explains_raw(w, g, raw)) {
                                k = kk;
                                ++outside;
                            }
                        }
                        if (k < 32) {
                            hdr[ALIAS_OFF + k] += 1;
                        } else {
                            ++other;
                        }
                    }
                    // Rewriting the fill data keeps every page correct, so a write that lands on another page
                    // shows up there as an alias on its next read.
                    if (writeback) {
                        for (uint32_t i = 0; i < n; ++i) {
                            noc_async_write(gold_addr + i * PAGE_BYTES, buf.get_noc_addr(c0 + i), PAGE_BYTES);
                        }
                        noc_async_write_barrier();
                    }
                }
            }
            ++pass;
        }
        hdr[1] = pass;
        hdr[2] = bad_pages;
        hdr[3] = bad_words;
        hdr[4] = stale;
        hdr[5] = other;
        hdr[6] = undecoded;
        hdr[13] = outside;
    }
    hdr[0] = seed;
    hdr[15] = OUT_MAGIC;
    noc_async_write(hdr_addr, out.get_noc_addr(slot), (ROW_WORDS * 4));
    noc_async_write_barrier();
}
