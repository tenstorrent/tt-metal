// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// K-delivery schedule shared by the reader and the compute K-walk (Perf 2, perf_experiments/kwalk_head_ramp).
//
// Without the ramp the K-walk delivers K as [block_k x (num_k_blocks - 1), last_block_k]: compute idles until a
// whole block_k-wide K-block of BOTH operands has been read and (R1a) multicast -- PERF FOCUS 1: 3.45 MB
// chip-wide, the first K-block lands 15.7 us after kernel start, the K-walk's entire exposed data movement.
// GM_K_RAMP = 1 carves a head ramp off the front of K: deliveries round(block_k * {2, 3, 4, 5} / 6) (block_k 12:
// 4, 6, 8, 10 tiles, stopping before the head would cover all of K), then whole block_k deliveries, then the
// remainder last. Each delivery lands before compute finishes the previous one (delivery time grows faster than
// linearly in its width), so compute starts after a third of the bytes and never waits again. Measured: PERF
// FOCUS 1 418.2 -> 409.6 us, 384x1536x1024 bias 35.6 -> 33.1 us, 8192x4096x64 (R1) 291.2 -> 282.2 us,
// bit-identical on the raw walk. (Moving the ragged block first instead was measured NULL: the next full block
// lands too late behind it.)
//
// Every delivery still occupies one nominal CB slot (block_m x block_k A, block_k x block_n B); push / pop counts
// are nominal. GM_COMPACT_A = 1 (raw K-walk only: compute_kernel_lib::matmul_block strides A rows by block_k) lays
// A out with row stride kv (the delivery's K width), so an R1a line injector multicasts only block_m * kv A tiles.
// B's slot is K-row-major, so the valid tiles of a short delivery are always a prefix (kv * block_n).
// With GM_K_RAMP = 0 the schedule is exactly the legacy [block_k.., last_block_k] (R2 / R3 / R4, R1b, residency,
// one-K-block walks, and the R1a helper walk -- see the host's k_head_ramp()).
#include <cstdint>

#ifndef GM_K_RAMP
#define GM_K_RAMP 0
#endif
#ifndef GM_COMPACT_A
#define GM_COMPACT_A 0
#endif

namespace gm_ks {

constexpr bool ramp = GM_K_RAMP != 0;
constexpr bool compact_a = GM_COMPACT_A != 0;
static_assert(!compact_a || ramp);

template <uint32_t bk>
constexpr uint32_t head_at(uint32_t i) {
    if (!ramp || i >= 4) {
        return 0;
    }
    const uint32_t h = (bk * (i + 2) + 3) / 6;
    return h == 0 ? 1 : h;
}

template <uint32_t bk, uint32_t nkb, uint32_t last>
struct Sched {
    static constexpr uint32_t tk = (nkb - 1) * bk + last;
    static constexpr uint32_t count_head() {
        uint32_t n = 0, s = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t h = head_at<bk>(i);
            if (h == 0 || s + h >= tk) {
                break;
            }
            s += h;
            ++n;
        }
        return n;
    }
    static constexpr uint32_t hn = count_head();
    static constexpr uint32_t head_prefix(uint32_t i) {
        uint32_t s = 0;
        for (uint32_t j = 0; j < i && j < hn; ++j) {
            s += head_at<bk>(j);
        }
        return s;
    }
    static constexpr uint32_t hs = head_prefix(hn);
    static constexpr uint32_t body = (tk - hs) / bk;
    static constexpr uint32_t rem = (tk - hs) % bk;
    // Number of K deliveries (== nkb without the ramp).
    static constexpr uint32_t n = hn + body + (rem != 0 ? 1 : 0);
    static_assert(ramp || n == nkb, "the legacy schedule must be the identity");
    static_assert(head_at<bk>(3) <= bk, "a head delivery must fit one slot");

    // K width of delivery i.
    static inline __attribute__((always_inline)) uint32_t kv(uint32_t i) {
        if (i < hn) {
            return head_at<bk>(i);
        }
        return (i - hn) < body ? bk : rem;
    }
    // First K tile of delivery i.
    static inline __attribute__((always_inline)) uint32_t k0(uint32_t i) {
        return i < hn ? head_prefix(i) : hs + (i - hn) * bk;
    }
};

}  // namespace gm_ks
