// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace sdpa::streaming {

struct StateTransfer {
    // Save / Restore move a whole Q block's state while compute waits. The streamed actions (ring kernels with
    // fused chunks, SDPA_RING_STREAM_STATE) overlap it with compute: SaveRows moves the O rows [Row0, Row0 + Rows)
    // as the last K chunk finishes them, SaveTail the remaining O rows from Row0 plus maxima, sums and header;
    // RestoreStream acks once for header, maxima and sums, then once per O row as it lands. A SaveTail with
    // Rows = 1 acks once its bytes have left L1 (the next block reuses the banks without a restore).
    enum Action : uint32_t { Save, Restore, SaveRows, SaveTail, RestoreStream };
    enum Word : uint32_t { Operation, Slot, Numerator, Maximum, Denominator, Chunks, Row0, Rows, Words };
    static constexpr uint32_t page_bytes = 4096;

    // Raw bytes of one Q block's state planes, for q_tiles query tile rows: the FP32 numerator (the BF16
    // recipes' CB also holds per-chunk scratch beside it, which is not transferred), BF16 maxima, and the
    // FP32 denominator.
    template <bool fp32, uint32_t q_tiles, uint32_t d_tiles = 4>
    static constexpr uint32_t plane_bytes(uint32_t plane) {
        return plane == 0 ? q_tiles * d_tiles * 4096 : plane == 1 ? q_tiles * 2048 : q_tiles * 4096;
    }

    // Transfer pages per plane. An odd Q tile count leaves the BF16 maxima plane half a page long; its
    // last page is transferred partially (plane_bytes % page_bytes bytes), never past the plane.
    template <bool fp32, uint32_t q_tiles, uint32_t d_tiles = 4>
    static constexpr uint32_t plane_pages(uint32_t plane) {
        return (plane_bytes<fp32, q_tiles, d_tiles>(plane) + page_bytes - 1) / page_bytes;
    }

    // Host-side page count for a runtime Q chunk; matches pages<fp32, q_tiles>.
    static constexpr uint32_t page_count(bool /*fp32*/, uint32_t q_tiles, uint32_t d_tiles = 4) {
        const uint32_t numerator = q_tiles * d_tiles * 4096;
        const uint32_t maxima = q_tiles * 2048;
        const uint32_t denominator = q_tiles * 4096;
        const auto ceil_pages = [](uint32_t bytes) { return (bytes + page_bytes - 1) / page_bytes; };
        return ceil_pages(numerator) + ceil_pages(maxima) + ceil_pages(denominator);
    }

    template <bool fp32, uint32_t q_tiles, uint32_t d_tiles = 4>
    static constexpr uint32_t pages = plane_pages<fp32, q_tiles, d_tiles>(0) +
                                      plane_pages<fp32, q_tiles, d_tiles>(1) + plane_pages<fp32, q_tiles, d_tiles>(2);

    template <bool fp32, uint32_t q_tiles, uint32_t d_tiles = 4>
    static constexpr bool page_aligned = plane_bytes<fp32, q_tiles, d_tiles>(0) % page_bytes == 0 &&
                                         plane_bytes<fp32, q_tiles, d_tiles>(1) % page_bytes == 0 &&
                                         plane_bytes<fp32, q_tiles, d_tiles>(2) % page_bytes == 0;
};

}  // namespace sdpa::streaming
