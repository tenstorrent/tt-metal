// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — per-image work geometry shared by reader, compute and writer.
//
// A core's block is `H_block` tile-rows x `K` tile-columns of the flattened (N*HW, C)
// tensor. Which tile-rows belong to image `n` differs by placement:
//   interleaved : the host hands every core the same tile-row range [r0, r0 + H_core) of
//                 EVERY image, so image n's sub-block is the whole block. Every stick of every
//                 tile-row is valid except, when HW % 32 != 0, the image's LAST
//                 tile-row, which carries only `row_hi = HW - 32*(HWt-1)` valid sticks — the host
//                 hands the core that owns it `row_hi`, every other core 32.
//   block-shard : the core's block is its resident shard — sticks [s0, s0 + sticks_valid) of
//                 the flattened tensor. Image n owns the intersection with
//                 [n*hw_stride, n*hw_stride + HW), where hw_stride is the stick pitch between
//                 images: HW for ROW_MAJOR (no padding between images), HWt*32 for TILE (each
//                 image is padded to whole tile-rows). For ROW_MAJOR shards that intersection
//                 may start / end mid tile-row (the shard height is a stick count), for TILE
//                 shards it ends early when HW % 32 != 0, so the first and last tile-rows carry
//                 the valid stick sub-range [lo, 32) / [0, hi). A core whose shard misses image n
//                 has no work in it (but still takes part in the stats combine — see writer).
//
// All three kernels evaluate this ONCE per image from the same runtime args, so the
// tile-row offsets, the staged-stick zeroing, the pass-2 row masks and the valid-stick
// output writes agree by construction.

#pragma once

#include <cstdint>

namespace groupnorm_geometry {

struct ImageWork {
    uint32_t row_off;  // first tile-row of image n's sub-block inside the core's block
    uint32_t rows;     // tile-rows in the sub-block (0 -> no work in this image)
    uint32_t lo;       // first valid stick of tile-row `row_off`            (0..31)
    uint32_t hi;       // one past the last valid stick of the last tile-row (1..32)
    bool active;       // rows > 0
};

// Interleaved placement: the same [0, H_core) sub-block for every image; the last tile-row of
// the block holds `row_hi` valid sticks (32 unless this core owns the image's ragged tail row).
inline ImageWork image_work_interleaved(uint32_t H_core, uint32_t row_hi) {
    return ImageWork{0, H_core, 0, row_hi, H_core > 0};
}

// Block-sharded placement: intersection of the core's sticks [s0, s0 + sticks_valid) with
// image n's sticks [n*hw_stride, n*hw_stride + HW), expressed in shard-local tile-rows.
inline ImageWork image_work_sharded(
    uint32_t image, uint32_t s0, uint32_t sticks_valid, uint32_t HW, uint32_t hw_stride) {
    const uint32_t img_lo = image * hw_stride;
    const uint32_t img_hi = img_lo + HW;
    const uint32_t blk_hi = s0 + sticks_valid;
    const uint32_t lo = img_lo > s0 ? img_lo : s0;
    const uint32_t hi = img_hi < blk_hi ? img_hi : blk_hi;
    if (lo >= hi) {
        return ImageWork{0, 0, 0, 32, false};
    }
    const uint32_t a = lo - s0;  // local sticks [a, b)
    const uint32_t b = hi - s0;
    const uint32_t r0 = a >> 5;
    const uint32_t r1 = (b + 31) >> 5;
    return ImageWork{r0, r1 - r0, a - (r0 << 5), b - ((r1 - 1) << 5), true};
}

// Valid stick range [sa, sb) of the i-th tile-row (0-based inside the sub-block).
inline void row_sticks(const ImageWork& w, uint32_t i, uint32_t& sa, uint32_t& sb) {
    sa = (i == 0) ? w.lo : 0;
    sb = (i + 1 == w.rows) ? w.hi : 32;
}

// Pass-2 row masks (`hw_mask` programs): a tile-row with invalid sticks must not let its zero
// pad sticks contribute (0 - mean)^2. The head row is masked when it starts late (or is the
// only row and ends early); the tail row when it ends early. Writer builds and pushes the
// masks in this order (head, then tail); compute pops them in the same order.
inline bool head_masked(const ImageWork& w) { return w.active && (w.lo != 0 || (w.rows == 1 && w.hi != 32)); }
inline bool tail_masked(const ImageWork& w) { return w.active && w.rows > 1 && w.hi != 32; }

// Pass-2 chunk sequence of an image: [head: 1 row] [body: Q-row chunks over [body_start,
// body_end)] [tail: 1 row]. Compute runs its masked-mean segments in this order and the
// streaming TILE reader pushes its nominal Q*K-page chunks in the SAME order, so the two agree
// on every chunk boundary (a body chunk that swallowed the tail row would be popped short).
struct Pass2Segments {
    bool head;
    bool tail;
    uint32_t body_start;
    uint32_t body_end;
};

inline Pass2Segments pass2_segments(const ImageWork& w, bool hw_mask) {
    const bool head = hw_mask && head_masked(w);
    const bool tail = hw_mask && tail_masked(w);
    return Pass2Segments{head, tail, head ? 1u : 0u, w.rows - (tail ? 1u : 0u)};
}

}  // namespace groupnorm_geometry
