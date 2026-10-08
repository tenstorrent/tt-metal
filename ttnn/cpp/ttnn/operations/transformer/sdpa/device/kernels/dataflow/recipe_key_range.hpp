// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// K-range + edge-mask model of the dense recipe reader (host: RecipeKeyRange in sdpa_recipe.hpp).
//
// Query row q (global position q_offset + local row) attends to one key interval [lo(q), hi(q)); neither end
// moves left as q grows. Causal, sliding windows (causal or centred), chunked-prefill offsets and windowed
// segments (cu_window_seqlens) all have this form. For a Q chunk the reader classifies each K chunk:
//   - outside the union of its rows' intervals: skipped (no reads, no compute);
//   - inside every row's interval: full, the recipe's unmasked chunk (fused for STANDARD / FAST);
//   - otherwise: edge, the recipe's additive-mask chunk, with {0, kRecipeMaskedBf16} tiles generated here.
// Compute learns [first, end) and the full sub-range [full_begin, full_end) from one control page per Q chunk.
//
// Masked keys add -2^100 rather than -inf: a row whose keys in a Q chunk's first K chunk are all masked then has
// a finite running max (s - m = 0, every P finite), which the next chunk's real keys replace with a zero
// rescale. With -inf that row would compute exp(-inf - -inf).
//
// The reader streams only the K chunks in range; the writer sends compute the control page and generates the edge
// masks (as legacy SDPA's writer generates windowed masks), so mask work never stalls the K/V stream.
//
// Compile-time switches (host: run_recipe_segments): SDPA_RECIPE_CAUSAL, SDPA_RECIPE_WINDOW (sliding window
// tokens, 0 = none), SDPA_RECIPE_SEGMENTS (cu_window_seqlens entries, windowed mode), SDPA_RECIPE_Q_SLAB_JOBS
// (ring-distributed Q slabs).

#include <cstdint>

// Each dataflow kernel's half of the scratch CB (the writer's starts SDPA_RECIPE_SCRATCH_WRITER bytes in): a slot for
// the Q offset, then cu_window_seqlens. The reader's page-table row follows both halves
// (SDPA_RECIPE_PAGE_TABLE_OFFSET).
struct RecipeScratch {
    static constexpr uint32_t Offset = 0;
    static constexpr uint32_t Segments = 128;
};

// Read one page of a row-major int32 tensor (Q offset, page-table row, cu_window_seqlens) to `address`; returns its
// first value.
template <typename Accessor>
FORCE_INLINE uint32_t
recipe_read_index_page(const Noc& noc, const Accessor& tensor, uint32_t page, uint32_t bytes, uint32_t address) {
    noc.async_read(tensor, CoreLocalMem<uint32_t>(address), bytes, {.page_id = page}, {});
    noc.async_read_barrier();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
}

// Masks are BFP4 tiles (576 B, a quarter of BF16, so an edge row group fits next to the fused chunks' CBs): 64
// shared exponents (one per 16-value face row), then 4-bit sign-magnitude mantissas, face by face. With every
// exponent at 2^100, mantissa 0 is 0 and 0xC (sign, 1.00) is -2^100: both exact.
constexpr uint32_t kRecipeMaskTileBytes = 576;
constexpr uint32_t kRecipeMaskExponents = 0xE3E3E3E3;  // 127 + 100
constexpr uint32_t kRecipeMaskedNibbles = 0xCCCCCCCC;

// A core's Q chunks are a contiguous range of one head's zigzag order 0, J-1, 1, J-2, ...: under a causal mask
// chunk j costs about j + 1 K chunks, so consecutive pairs cost the same (host: recipe_zigzag_split).
FORCE_INLINE uint32_t recipe_zigzag_job(uint32_t z, uint32_t jobs_per_head) {
    return z % 2 == 0 ? z / 2 : jobs_per_head - 1 - z / 2;
}

#ifdef SDPA_RECIPE_Q_SLAB_JOBS
// Ring-distributed SDPA (host: RecipeKeyRange::q_slab_rows): a head's Q chunks are two slabs of
// SDPA_RECIPE_Q_SLAB_JOBS whole chunks of the sequence, starting at its chunks first[0] and first[1].
struct RecipeQSlabs {
    uint32_t first[2];
    FORCE_INLINE uint32_t chunk(uint32_t job) const {
        return job < SDPA_RECIPE_Q_SLAB_JOBS ? first[0] + job : first[1] + job - SDPA_RECIPE_Q_SLAB_JOBS;
    }
};
#endif

struct RecipeChunkRange {
    uint32_t first, end;            // K chunks to process
    uint32_t full_begin, full_end;  // the ones every row sees whole (no mask)
};

struct RecipeKeyRange {
    uint32_t q_offset;  // global position of local Q row 0
    uint32_t k_rows;    // logical K length
#if SDPA_RECIPE_SEGMENTS > 0
    const volatile tt_l1_ptr uint32_t* segments;  // cu_window_seqlens: SDPA_RECIPE_SEGMENTS cumulative bounds
#endif

    // Valid keys [lo, hi) of global query position q.
    FORCE_INLINE void keys(uint32_t q, uint32_t& lo, uint32_t& hi) const {
        lo = 0;
        hi = k_rows;
#if SDPA_RECIPE_SEGMENTS > 0
        // The window holding q: the last bound <= q. Rows past the last window see nothing (lo = hi = k_rows).
        if (q >= segments[SDPA_RECIPE_SEGMENTS - 1]) {
            lo = k_rows;
            return;
        }
        uint32_t a = 0, b = SDPA_RECIPE_SEGMENTS - 1;
        while (b - a > 1) {
            const uint32_t m = (a + b) / 2;
            if (segments[m] <= q) {
                a = m;
            } else {
                b = m;
            }
        }
        lo = segments[a];
        hi = hi < segments[b] ? hi : segments[b];
#endif
#if SDPA_RECIPE_CAUSAL
        hi = hi < q + 1 ? hi : q + 1;
#endif
#if SDPA_RECIPE_WINDOW > 0
#if SDPA_RECIPE_CAUSAL
        const uint32_t window_lo = q + 1 >= SDPA_RECIPE_WINDOW ? q + 1 - SDPA_RECIPE_WINDOW : 0;
#else
        constexpr uint32_t half = SDPA_RECIPE_WINDOW / 2;
        const uint32_t window_lo = q >= half ? q - half : 0;
        hi = hi < q + half + 1 ? hi : q + half + 1;
#endif
        lo = lo > window_lo ? lo : window_lo;
#endif
        if (lo > hi) {
            lo = hi;
        }
    }

    // K chunks of the Q chunk whose valid local rows are [row0, row_end). Never empty: a Q chunk without any
    // visible key still runs one (fully masked) chunk, so every kernel's per-chunk protocol stays the same.
    FORCE_INLINE RecipeChunkRange
    chunks(uint32_t row0, uint32_t row_end, uint32_t k_chunk_rows, uint32_t k_chunks) const {
        uint32_t lo_first, hi_first, lo_last, hi_last;
        keys(q_offset + row0, lo_first, hi_first);
        keys(q_offset + row_end - 1, lo_last, hi_last);
        RecipeChunkRange r;
        r.first = lo_first / k_chunk_rows;
        r.end = (hi_last + k_chunk_rows - 1) / k_chunk_rows;
        if (r.first >= k_chunks) {
            r.first = k_chunks - 1;
        }
        if (r.end <= r.first) {
            r.end = r.first + 1;
        }
        r.full_begin = (lo_last + k_chunk_rows - 1) / k_chunk_rows;
        r.full_end = hi_first / k_chunk_rows;
        if (r.full_end < r.full_begin || hi_first <= lo_last) {
            r.full_end = r.full_begin;
        }
        return r;
    }

    // Mask tile for query rows [q0, q0 + 32) and keys [k0, k0 + 32): 0 (all visible), 1 (all masked) or 2 (mixed).
    FORCE_INLINE uint32_t tile_kind(uint32_t q0, uint32_t k0) const {
        uint32_t lo_first, hi_first, lo_last, hi_last;
        keys(q0, lo_first, hi_first);
        keys(q0 + 31, lo_last, hi_last);
        if (lo_last <= k0 && hi_first >= k0 + 32) {
            return 0;
        }
        if (hi_last <= k0 || lo_first >= k0 + 32) {
            return 1;
        }
        return 2;
    }

    // A mixed tile's pattern depends only on q0 - k0 for causal and sliding-window rows away from the K tail; the
    // reader then reuses one generated copy (returns false for windowed segments and the K tail).
    FORCE_INLINE bool tile_repeats([[maybe_unused]] uint32_t k0) const {
#if SDPA_RECIPE_SEGMENTS > 0
        return false;
#else
        return k0 + 32 <= k_rows;
#endif
    }

    // Write a mixed BFP4 mask tile: 0 where visible, -2^100 elsewhere.
    FORCE_INLINE void write_tile(uint32_t q0, uint32_t k0, uint32_t address) const {
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
        for (uint32_t i = 0; i < 16; ++i) {
            words[i] = kRecipeMaskExponents;
        }
        for (uint32_t row = 0; row < 32; ++row) {
            uint32_t lo, hi;
            keys(q0 + row, lo, hi);
            // Visible columns [a, b) of this tile row.
            const uint32_t a = lo <= k0 ? 0 : lo - k0 < 32 ? lo - k0 : 32;
            const uint32_t b = hi <= k0 ? 0 : hi - k0 < 32 ? hi - k0 : 32;
            // A face is 32 words of mantissas (two per 16-value row, low nibble first); a tile row spans faces
            // (row / 16) * 2 and + 1.
            volatile tt_l1_ptr uint32_t* face_row = words + 16 + (row / 16) * 64 + (row % 16) * 2;
            for (uint32_t group = 0; group < 4; ++group) {
                uint32_t nibbles = 0;
                for (uint32_t i = 0; i < 8; ++i) {
                    const uint32_t c = group * 8 + i;
                    nibbles |= c >= a && c < b ? 0 : 0xCu << (4 * i);
                }
                face_row[(group / 2) * 32 + group % 2] = nibbles;
            }
        }
    }
};

#ifdef SDPA_RECIPE_MASK_CB
// Generated mixed mask tiles kept for reuse (RecipeKeyRange::tile_repeats), after the all-masked template in
// SDPA_RECIPE_MASKED_TILE_CB (host: kRecipeMaskCacheTiles). A causal call needs one (the diagonal), a sliding window
// two or three.
constexpr uint32_t kRecipeMaskCacheTiles = SDPA_RECIPE_MASK_CACHE_TILES;
struct RecipeMaskCache {
    uint32_t base;  // the all-masked template; slot i at base + (i + 1) tiles
    int32_t delta[kRecipeMaskCacheTiles];
    uint32_t used = 0;
    uint32_t next = 0;  // round-robin eviction once full
};

FORCE_INLINE void copy_local_tile(const Noc& noc, uint32_t source, uint32_t destination) {
    const uint8_t noc_id = noc.get_noc_id();
    UnicastEndpoint self;
    noc.async_read(
        self,
        CoreLocalMem<uint32_t>(destination),
        kRecipeMaskTileBytes,
        {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = source},
        {});
}

// An edge K chunk's mask, in the attn_mask path's CB and row-group order (reader_recipe.cpp: read_mask_chunk):
// all-visible tiles are zero-filled, all-masked ones copied from the template tile, mixed ones copied from the cache or
// written row by row.
template <uint32_t q_tiles>
FORCE_INLINE void generate_mask_chunk(
    const Noc& noc,
    const RecipeKeyRange& keys,
    CircularBuffer& cb,
    RecipeMaskCache& cache,
    uint32_t q_row0,
    uint32_t k_row0) {
    constexpr uint32_t bytes = get_tile_size(SDPA_RECIPE_MASK_CB);
    static_assert(bytes == kRecipeMaskTileBytes, "K-range masks are BFP4");
    constexpr uint32_t rows =
        (q_tiles + SDPA_RECIPE_MASK_GROUP_ROWS - 1) / SDPA_RECIPE_MASK_GROUP_ROWS * SDPA_RECIPE_MASK_GROUP_ROWS;
    for (uint32_t row = 0; row < rows; ++row) {
        cb.reserve_back(SDPA_K_CHUNK_TILES);
        const uint32_t ptr = cb.get_write_ptr();
        bool zeroed = false;
        for (uint32_t col = 0; col < SDPA_K_CHUNK_TILES; ++col) {
            const uint32_t address = ptr + col * bytes;
            const uint32_t q0 = keys.q_offset + q_row0 + row * 32, k0 = k_row0 + col * 32;
            const uint32_t kind = row < q_tiles ? keys.tile_kind(q0, k0) : 0;
            if (kind == 0) {
                noc.async_write_zeros(CoreLocalMem<uint32_t>(address), bytes);
                zeroed = true;
            } else if (kind == 1) {
                copy_local_tile(noc, cache.base, address);
            } else if (!keys.tile_repeats(k0)) {
                keys.write_tile(q0, k0, address);
            } else {
                const int32_t delta = static_cast<int32_t>(q0 - k0);
                uint32_t slot = 0;
                while (slot < cache.used && cache.delta[slot] != delta) {
                    ++slot;
                }
                if (slot == cache.used) {
                    // Miss: evict round-robin once full; copies still reading the slot must land first.
                    if (cache.used < kRecipeMaskCacheTiles) {
                        ++cache.used;
                    } else {
                        slot = cache.next;
                        cache.next = (cache.next + 1) % kRecipeMaskCacheTiles;
                    }
                    noc.async_read_barrier();
                    keys.write_tile(q0, k0, cache.base + (slot + 1) * bytes);
                    cache.delta[slot] = delta;
                }
                copy_local_tile(noc, cache.base + (slot + 1) * bytes, address);
            }
        }
        if (zeroed) {
            noc.write_zeros_l1_barrier();
        }
        noc.async_read_barrier();
        cb.push_back(SDPA_K_CHUNK_TILES);
    }
}
#endif

// The control page compute reads per Q chunk (recipe_streaming.hpp: recipe_read_key_range).
FORCE_INLINE void recipe_push_chunk_range(CircularBuffer& cb, const RecipeChunkRange& range) {
    cb.reserve_back(1);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    words[0] = range.first;
    words[1] = range.end;
    words[2] = range.full_begin;
    words[3] = range.full_end;
    cb.push_back(1);
}
