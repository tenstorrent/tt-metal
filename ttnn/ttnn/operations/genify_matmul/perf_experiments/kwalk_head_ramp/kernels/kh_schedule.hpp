// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// kwalk_head_ramp (perf experiment): the K-delivery schedule shared by the reader and the compute K-walk.
//
// The op today delivers K as [block_k x (num_k_blocks - 1), last_block_k] -- the first delivery is a full
// K-block, so compute idles until a whole block_k-wide K-block of BOTH operands has been read by the line
// injectors and multicast (PERF FOCUS 1: 3.45 MB chip-wide, 15.7 us). This schedule lets compute start on
// a SMALL first delivery:
//   KH_H0..KH_H3   head deliveries (K tiles each, <= block_k), carved off the front of K
//   KH_RAGGED_FIRST  the remainder (tensor_k - head) % block_k goes FIRST instead of last
// then whole block_k deliveries, then the remainder (unless it went first). No defines = the op's schedule.
//   ragged_first (focus): [4, 12 x 13]      ramp KH_H0=4 KH_H1=8 (focus): [4, 8, 12 x 12, 4]
// Every delivery still occupies one nominal CB slot (block_m x block_k A, block_k x block_n B).
//   KH_COMPACT_A   A is laid out with row stride kv (the delivery's K width) instead of block_k, and the
//                  injector multicasts only block_m * kv tiles (the raw K-walk reads it with that stride).
//   KH_COMPACT_B   the injector multicasts only the kv * block_n prefix of the B slot (its layout is
//                  already K-row-major, so the valid tiles of a short delivery are a prefix).
#include <cstdint>

#ifndef KH_H0
#define KH_H0 0
#endif
#ifndef KH_H1
#define KH_H1 0
#endif
#ifndef KH_H2
#define KH_H2 0
#endif
#ifndef KH_H3
#define KH_H3 0
#endif
#ifndef KH_RAGGED_FIRST
#define KH_RAGGED_FIRST 0
#endif
// KH_RAMP_AUTO: the head is the linear ramp round(block_k * {2, 3, 4, 5} / 6) (focus block_k 12: 4, 6, 8, 10),
// overriding KH_H0..KH_H3 -- the measured-best focus ramp expressed relative to block_k.
#ifndef KH_RAMP_AUTO
#define KH_RAMP_AUTO 0
#endif
#ifndef KH_COMPACT_A
#define KH_COMPACT_A 0
#endif
#ifndef KH_COMPACT_B
#define KH_COMPACT_B 0
#endif

namespace kh {

constexpr uint32_t head_manual(uint32_t i) {
    return i == 0 ? KH_H0 : i == 1 ? KH_H1 : i == 2 ? KH_H2 : i == 3 ? KH_H3 : 0;
}
template <uint32_t bk>
constexpr uint32_t head_at(uint32_t i) {
    if constexpr (KH_RAMP_AUTO != 0) {
        if (i >= 4) {
            return 0;
        }
        const uint32_t h = (bk * (i + 2) + 3) / 6;
        return h == 0 ? 1 : h;
    } else {
        return head_manual(i);
    }
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
    static constexpr uint32_t rest = tk - hs;
    static constexpr uint32_t body = rest / bk;
    static constexpr uint32_t rem = rest % bk;
    static constexpr bool rf = KH_RAGGED_FIRST != 0 && rem != 0;
    static constexpr uint32_t n = hn + body + (rem != 0 ? 1 : 0);
    static_assert(
        head_at<bk>(0) <= bk && head_at<bk>(1) <= bk && head_at<bk>(2) <= bk && head_at<bk>(3) <= bk,
        "a head delivery must fit one slot");

    // K width of delivery i.
    static inline __attribute__((always_inline)) uint32_t kv(uint32_t i) {
        if (i < hn) {
            return head_at<bk>(i);
        }
        const uint32_t j = i - hn;
        if constexpr (rf) {
            return j == 0 ? rem : bk;
        } else {
            return j < body ? bk : rem;
        }
    }
    // First K tile of delivery i.
    static inline __attribute__((always_inline)) uint32_t k0(uint32_t i) {
        if (i < hn) {
            return head_prefix(i);
        }
        const uint32_t j = i - hn;
        if constexpr (rf) {
            return j == 0 ? hs : hs + rem + (j - 1) * bk;
        } else {
            return hs + j * bk;
        }
    }
};

}  // namespace kh
