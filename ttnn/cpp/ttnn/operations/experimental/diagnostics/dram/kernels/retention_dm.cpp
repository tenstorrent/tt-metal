// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GDDR retention sweep: fill, check and poke modes for test_gddr_retention
// (tests/dram/test_dram_retention.py).
// The RETENTION_* macros are defines the test passes from its own constants (DM_DEFINES),
// so this kernel and the host-side report decoder share one definition.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE_BYTES = RETENTION_PAGE_BYTES;
constexpr uint32_t PAGE_WORDS = RETENTION_PAGE_WORDS;

FORCE_INLINE uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

// One hash per 64 B line keeps generation cheap. Seeds 2m and 2m + 1 hash alike and the odd one is inverted.
FORCE_INLINE uint32_t line_key(uint32_t addr, uint32_t bank, uint32_t seed) {
    return mix(mix((addr >> 6) ^ ((seed >> 1) * 0x9E3779B9u)) ^ (bank * 0x85EBCA77u));
}

FORCE_INLINE uint32_t pattern_word(uint32_t addr, uint32_t bank, uint32_t seed) {
    const uint32_t value = line_key(addr, bank, seed) ^ (((addr >> 2) & 15) * 0x01234567u);
    return (seed & 1) ? ~value : value;
}

FORCE_INLINE void fill_page(
    volatile tt_l1_ptr uint32_t* dst, uint32_t addr, uint32_t words, uint32_t bank, uint32_t seed) {
    const uint32_t invert = (seed & 1) ? 0xFFFFFFFFu : 0;
    for (uint32_t line = 0; line < (words >> 4); ++line) {
        const uint32_t key = line_key(addr + 64 * line, bank, seed) ^ invert;
        for (uint32_t word = 0; word < 16; ++word) {
            dst[16 * line + word] = key ^ (word * 0x01234567u);
        }
    }
}

// OR of (read ^ expected) over one 64 B line, with the expected words rebuilt in registers from the line key.
FORCE_INLINE uint32_t line_diff(const tt_l1_ptr uint32_t* observed, uint32_t key) {
    uint32_t diff = 0;
#pragma GCC unroll 16
    for (uint32_t word = 0; word < 16; ++word) {
        diff |= observed[word] ^ key ^ (word * 0x01234567u);
    }
    return diff;
}

FORCE_INLINE uint32_t byte_mask(uint32_t x) {
    uint32_t mask = 0;
    for (uint32_t byte = 0; byte < 4; ++byte) {
        mask |= (x & (0xFFu << (8 * byte))) ? (0xFFu << (8 * byte)) : 0;
    }
    return mask;
}

FORCE_INLINE uint32_t bit_count(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

// True when every wrong byte of the page equals the pattern written with `seed` at address ^ flip.
FORCE_INLINE bool explains(
    volatile tt_l1_ptr uint32_t* observed,
    volatile tt_l1_ptr uint32_t* expected,
    uint32_t addr,
    uint32_t words,
    uint32_t bank,
    uint32_t seed,
    uint32_t flip) {
    for (uint32_t word = 0; word < words; ++word) {
        const uint32_t diff = observed[word] ^ expected[word];
        if (diff && ((observed[word] ^ pattern_word((addr + 4 * word) ^ flip, bank, seed)) & byte_mask(diff))) {
            return false;
        }
    }
    return true;
}

// Same test against data read back from DRAM, for alias sources outside the covered range.
FORCE_INLINE bool explains_raw(
    volatile tt_l1_ptr uint32_t* observed,
    volatile tt_l1_ptr uint32_t* expected,
    volatile tt_l1_ptr uint32_t* reference,
    uint32_t words) {
    for (uint32_t word = 0; word < words; ++word) {
        const uint32_t diff = observed[word] ^ expected[word];
        if (diff && ((observed[word] ^ reference[word]) & byte_mask(diff))) {
            return false;
        }
    }
    return true;
}

// True when every 64 B line of the page, moved to line ^ flip, lies inside the covered range.
FORCE_INLINE bool source_covered(
    uint32_t addr, uint32_t words, uint32_t flip, uint32_t covered_lo, uint32_t covered_hi) {
    for (uint32_t line = 0; line < (words >> 4); ++line) {
        const uint32_t src = (addr + 64 * line) ^ flip;
        if (src < covered_lo || src >= covered_hi) {
            return false;
        }
    }
    return true;
}

// Reads what each 64 B line of the page holds at line ^ flip into `raw_addr`, line by line: a page can cross the
// flipped bit, so its source need not be one block. False if a source line is in the register space past DRAM_TOP.
FORCE_INLINE bool read_source(uint32_t raw_addr, uint32_t addr, uint32_t words, uint32_t bank, uint32_t flip) {
    bool readable = true;
    for (uint32_t line = 0; readable && line < (words >> 4); ++line) {
        const uint32_t src = (addr + 64 * line) ^ flip;
        readable = src < static_cast<uint32_t>(RETENTION_DRAM_TOP);
        if (readable) {
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, src), raw_addr + 64 * line, 64);
        }
    }
    noc_async_read_barrier();
    return readable;
}

void kernel_main() {
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t lo = get_arg_val<uint32_t>(1);
    const uint32_t hi = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t slot = get_arg_val<uint32_t>(4);
    const uint32_t seed = get_arg_val<uint32_t>(5);
    const uint32_t prev_seed = get_arg_val<uint32_t>(6);
    const uint32_t reps = get_arg_val<uint32_t>(7);
    const uint32_t pass_index = get_arg_val<uint32_t>(8);
    const uint32_t inject_addr = get_arg_val<uint32_t>(9);
    const uint32_t inject_bit = get_arg_val<uint32_t>(10);
    const uint32_t inject_mask = get_arg_val<uint32_t>(11);
    const uint32_t decode_max = get_arg_val<uint32_t>(12);
    const uint32_t covered_lo = get_arg_val<uint32_t>(13);
    const uint32_t covered_hi = get_arg_val<uint32_t>(14);
    constexpr uint32_t scratch = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t mode = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t hdr_addr = get_write_ptr(scratch);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr_addr);
    const uint32_t gold_addr = hdr_addr + (RETENTION_ROW_WORDS * 4);
    volatile tt_l1_ptr uint32_t* gold = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gold_addr);
    const uint32_t work_addr = gold_addr + chunk * PAGE_BYTES;
    volatile tt_l1_ptr uint32_t* work = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(work_addr);
    const uint32_t raw_addr = work_addr + chunk * PAGE_BYTES;
    volatile tt_l1_ptr uint32_t* raw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_addr);
    volatile tt_l1_ptr uint32_t* failure_log = hdr + RETENTION_LOG_OFF;
    for (uint32_t word = 0; word < RETENTION_ROW_WORDS; ++word) {
        hdr[word] = 0;
    }

    if constexpr (mode == RETENTION_MODE_FILL) {
        for (uint32_t addr = lo; addr < hi;) {
            for (uint32_t page = 0; addr < hi && page < chunk; ++page) {
                const uint32_t bytes = (hi - addr < PAGE_BYTES) ? (hi - addr) : PAGE_BYTES;
                volatile tt_l1_ptr uint32_t* expected = gold + page * PAGE_WORDS;
                fill_page(expected, addr, bytes >> 2, bank, seed);
                if (inject_addr >= addr && inject_addr - addr < bytes) {
                    for (uint32_t word = 0; word < (bytes >> 2); ++word) {
                        const uint32_t alias = pattern_word((addr + 4 * word) ^ (1u << inject_bit), bank, seed);
                        expected[word] = (expected[word] & ~inject_mask) | (alias & inject_mask);
                    }
                }
                noc_async_write(gold_addr + page * PAGE_BYTES, get_noc_addr_from_bank_id<true>(bank, addr), bytes);
                addr += bytes;
            }
            noc_async_write_barrier();
        }
        hdr[RETENTION_H_WORDS_LO] = (hi - lo) >> 2;
    } else if constexpr (mode == RETENTION_MODE_POKE) {
        // Changes DRAM while the data waits, for the negative test: flip bits of one word (value is the XOR mask),
        // overwrite one 64 B line with the pattern `seed` wrote at line ^ value, or copy the line at line ^ value.
        const uint32_t pokes = get_arg_val<uint32_t>(15);
        for (uint32_t entry = 0; entry < pokes; ++entry) {
            const uint32_t base = 16 + RETENTION_POKE_FIELDS * entry;
            const uint32_t kind = get_arg_val<uint32_t>(base);
            const uint32_t poke_bank = get_arg_val<uint32_t>(base + 1);
            const uint32_t addr = get_arg_val<uint32_t>(base + 2);
            const uint32_t value = get_arg_val<uint32_t>(base + 3);
            const uint32_t poke_seed = get_arg_val<uint32_t>(base + 4);
            const uint32_t line = addr & ~63u;
            const uint64_t line_noc_addr = get_noc_addr_from_bank_id<true>(poke_bank, line);
            if (kind == RETENTION_POKE_FLIP) {
                noc_async_read(line_noc_addr, gold_addr, 64);
                noc_async_read_barrier();
                const uint32_t word = (addr & 63) >> 2;
                gold[word] = gold[word] ^ value;
            } else if (kind == RETENTION_POKE_COPY) {
                noc_async_read(get_noc_addr_from_bank_id<true>(poke_bank, line ^ value), gold_addr, 64);
                noc_async_read_barrier();
            } else {
                for (uint32_t word = 0; word < 16; ++word) {
                    gold[word] = pattern_word((line + 4 * word) ^ value, poke_bank, poke_seed);
                }
            }
            noc_async_write(gold_addr, line_noc_addr, 64);
            noc_async_write_barrier();
        }
        hdr[RETENTION_H_WORDS_LO] = pokes;
    } else {
        const uint32_t invert = (seed & 1) ? 0xFFFFFFFFu : 0;
        uint32_t bad_pages = 0, bad_words = 0, stale = 0, other = 0, undecoded = 0, decoded = 0, outside = 0;
        uint32_t logged = 0, bits_lost = 0, bits_gained = 0;
        uint64_t words_checked = 0;
        for (uint32_t chunk_addr = lo; chunk_addr < hi;) {
            const uint32_t chunk_bytes =
                (hi - chunk_addr < chunk * PAGE_BYTES) ? (hi - chunk_addr) : chunk * PAGE_BYTES;
            const uint32_t pages = (chunk_bytes + PAGE_BYTES - 1) / PAGE_BYTES;
            for (uint32_t rep = 0; rep < reps; ++rep) {
                for (uint32_t page = 0; page < pages; ++page) {
                    const uint32_t addr = chunk_addr + page * PAGE_BYTES;
                    const uint32_t bytes = (hi - addr < PAGE_BYTES) ? (hi - addr) : PAGE_BYTES;
                    noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr), work_addr + page * PAGE_BYTES, bytes);
                }
                noc_async_read_barrier();
                // The barrier ends with a fence; this also keeps the compiler from hoisting the plain loads above it.
                asm volatile("" ::: "memory");
                for (uint32_t page = 0; page < pages; ++page) {
                    const uint32_t addr = chunk_addr + page * PAGE_BYTES;
                    const uint32_t words = ((hi - addr < PAGE_BYTES) ? (hi - addr) : PAGE_BYTES) >> 2;
                    const tt_l1_ptr uint32_t* page_words =
                        reinterpret_cast<const tt_l1_ptr uint32_t*>(work_addr + page * PAGE_BYTES);
                    uint32_t page_diff = 0;
                    for (uint32_t line = 0; line < (words >> 4); ++line) {
                        page_diff |= line_diff(page_words + 16 * line, line_key(addr + 64 * line, bank, seed) ^ invert);
                    }
                    words_checked += words;
                    if (page_diff == 0) {
                        continue;
                    }
                    // Slow path, only for a page that miscompared: log each bad word, then classify the page.
                    volatile tt_l1_ptr uint32_t* observed = work + page * PAGE_WORDS;
                    volatile tt_l1_ptr uint32_t* expected = gold + page * PAGE_WORDS;
                    fill_page(expected, addr, words, bank, seed);
                    const uint32_t page_first_entry = logged;
                    uint32_t page_bad = 0;
                    for (uint32_t word = 0; word < words; ++word) {
                        const uint32_t diff = observed[word] ^ expected[word];
                        if (diff == 0) {
                            continue;
                        }
                        for (uint32_t byte = 0; byte < 4; ++byte) {
                            hdr[RETENTION_H_LANES + byte] =
                                hdr[RETENTION_H_LANES + byte] + (((diff >> (8 * byte)) & 0xFFu) ? 1 : 0);
                        }
                        bits_lost += bit_count(diff & expected[word]);
                        bits_gained += bit_count(diff & observed[word]);
                        if (logged < RETENTION_LOG_CAP) {
                            failure_log[4 * logged] = addr + 4 * word;
                            failure_log[4 * logged + 1] = observed[word];
                            failure_log[4 * logged + 2] = expected[word];
                            failure_log[4 * logged + 3] = rep;
                        }
                        ++logged;
                        ++page_bad;
                    }
                    bad_words += page_bad;
                    ++bad_pages;
                    // Each logged word of the page carries the page's verdict above the read (rep) it failed on.
                    auto tag_page = [&](uint32_t verdict) {
                        for (uint32_t entry = page_first_entry; entry < logged && entry < RETENTION_LOG_CAP; ++entry) {
                            failure_log[4 * entry + 3] = failure_log[4 * entry + 3] | (verdict << 16);
                        }
                    };
                    if (decoded >= decode_max) {
                        ++undecoded;
                        tag_page(RETENTION_VERDICT_UNDECODED);
                        continue;
                    }
                    ++decoded;
                    if (prev_seed != static_cast<uint32_t>(RETENTION_NONE) &&
                        explains(observed, expected, addr, words, bank, prev_seed, 0)) {
                        ++stale;
                        tag_page(RETENTION_VERDICT_STALE);
                        continue;
                    }
                    uint32_t alias_bit = 2;
                    for (; alias_bit < 32; ++alias_bit) {
                        const uint32_t flip = 1u << alias_bit;
                        if (source_covered(addr, words, flip, covered_lo, covered_hi)) {
                            if (explains(observed, expected, addr, words, bank, seed, flip)) {
                                break;
                            }
                        } else if (
                            // Bits below 6 stay inside the line, which is covered; the rest keep reads 64 B aligned.
                            alias_bit >= 6 && read_source(raw_addr, addr, words, bank, flip) &&
                            explains_raw(observed, expected, raw, words)) {
                            ++outside;
                            break;
                        }
                    }
                    if (alias_bit < 32) {
                        hdr[RETENTION_H_ALIAS + alias_bit] = hdr[RETENTION_H_ALIAS + alias_bit] + 1;
                        tag_page(alias_bit);
                    } else {
                        ++other;
                        tag_page(RETENTION_VERDICT_OTHER);
                    }
                }
            }
            chunk_addr += chunk_bytes;
        }
        hdr[RETENTION_H_BAD_PAGES] = bad_pages;
        hdr[RETENTION_H_BAD_WORDS] = bad_words;
        hdr[RETENTION_H_STALE] = stale;
        hdr[RETENTION_H_OTHER] = other;
        hdr[RETENTION_H_UNDECODED] = undecoded;
        hdr[RETENTION_H_LOGGED] = logged;
        hdr[RETENTION_H_WORDS_LO] = static_cast<uint32_t>(words_checked);
        hdr[RETENTION_H_WORDS_HI] = static_cast<uint32_t>(words_checked >> 32);
        hdr[RETENTION_H_BITS_LOST] = bits_lost;
        hdr[RETENTION_H_BITS_GAINED] = bits_gained;
        hdr[RETENTION_H_OUTSIDE] = outside;
    }
    hdr[RETENTION_H_SEED] = seed;
    hdr[RETENTION_H_PASS] = pass_index;
    hdr[RETENTION_H_BANK] = bank;
    hdr[RETENTION_H_LO] = lo;
    hdr[RETENTION_H_HI] = hi;
    hdr[RETENTION_H_MAGIC] = RETENTION_MAGIC;
    noc_async_write(hdr_addr, out.get_noc_addr(slot), (RETENTION_ROW_WORDS * 4));
    noc_async_write_barrier();
}
