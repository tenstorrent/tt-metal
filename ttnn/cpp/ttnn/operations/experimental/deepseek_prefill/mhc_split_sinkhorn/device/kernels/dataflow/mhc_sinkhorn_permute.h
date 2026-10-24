// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The reader's half of the SFPU Sinkhorn (SFPU_SINKHORN): L1-to-L1 element permutations between the
// token-row tile the matmul produces ([t, e]: token t in row t, comb element e = 4 i + j in column e)
// and the layout mhc_sinkhorn_sfpu.h wants in dst (element e = dst vector e, lane t = token). In the
// fp32 tile, element (t, e) of the token-row form is word  RW(t) + e  and of the SFPU form word
//   (e >> 3) * 256 + ((e & 7) >> 1) * 64 + (t >> 3) * 16 + (t & 7) * 2 + (e & 1)
// (face e / 8; rows 4 ((e % 8) / 2) .. +3 of the face; columns of parity e % 2; lane t). The unused
// words of the SFPU tile are zero (exp(0) = 1, normalised among themselves, never read back) and
// the pad columns [16, 32) of the comb output are zero. 512 word copies each way on the RISC,
// overlapped with the compute kernel's pre/post section and the next tile's matmuls.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

namespace mhc_sinkhorn_permute_detail {
constexpr uint32_t CB_COMB = 4;   // c_4  out (token rows)
constexpr uint32_t CB_LOG = 5;    // c_5  comb logits, token rows (compute -> reader)
constexpr uint32_t CB_SKIN = 6;   // c_6  comb logits in the SFPU layout (reader -> compute)
constexpr uint32_t CB_SKOUT = 7;  // c_7  the Sinkhorn result in the SFPU layout (compute -> reader)
constexpr uint32_t TILE_BYTES = 4096;
constexpr uint32_t N_ELEM = 16;  // n * n comb elements
constexpr uint32_t T = 32;       // tokens a tile

// Both layouts walk with constant strides when the element e is the outer loop and the tokens go in blocks
// of 8 (tb = t / 8, k = t % 8):
//   token rows: word  e + 512 (tb / 2) + 128 (tb % 2) + 16 k
//   SFPU form:  word  (e / 8) 256 + ((e % 8) / 2) 64 + (e % 2) + 16 tb + 2 k
// so the loops below unroll to loads and stores at immediate offsets.
FORCE_INLINE uint32_t row_base(uint32_t e, uint32_t tb) { return e + ((tb >> 1) << 9) + ((tb & 1) << 7); }
FORCE_INLINE uint32_t sk_base(uint32_t e, uint32_t tb) {
    return ((e >> 3) << 8) + (((e & 7) >> 1) << 6) + (e & 1) + (tb << 4);
}
}  // namespace mhc_sinkhorn_permute_detail

// CB_LOG (one token-row logits tile) -> CB_SKIN (the SFPU layout, zeros elsewhere); pops CB_LOG.
FORCE_INLINE void mhc_sinkhorn_permute_in(const Noc& noc) {
    using namespace mhc_sinkhorn_permute_detail;
    CircularBuffer cb_log(CB_LOG), cb_skin(CB_SKIN);
    cb_skin.reserve_back(1);
    noc.async_write_zeros(cb_skin, TILE_BYTES, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    cb_log.wait_front(1);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_log.get_read_ptr());
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_skin.get_write_ptr());
#pragma GCC unroll 16
    for (uint32_t e = 0; e < N_ELEM; ++e) {
#pragma GCC unroll 4
        for (uint32_t tb = 0; tb < T / 8; ++tb) {
            volatile tt_l1_ptr uint32_t* s = src + row_base(e, tb);
            volatile tt_l1_ptr uint32_t* d = dst + sk_base(e, tb);
#pragma GCC unroll 8
            for (uint32_t k = 0; k < 8; ++k) {
                d[2 * k] = s[16 * k];
            }
        }
    }
    cb_log.pop_front(1);
    cb_skin.push_back(1);
}

// CB_SKOUT (the Sinkhorn result in the SFPU layout) -> CB_COMB (token rows, pad columns zero); pops CB_SKOUT.
FORCE_INLINE void mhc_sinkhorn_permute_out(const Noc& noc) {
    using namespace mhc_sinkhorn_permute_detail;
    CircularBuffer cb_skout(CB_SKOUT), cb_comb(CB_COMB);
    cb_comb.reserve_back(1);
    noc.async_write_zeros(cb_comb, TILE_BYTES, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    cb_skout.wait_front(1);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_skout.get_read_ptr());
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_comb.get_write_ptr());
#pragma GCC unroll 16
    for (uint32_t e = 0; e < N_ELEM; ++e) {
#pragma GCC unroll 4
        for (uint32_t tb = 0; tb < T / 8; ++tb) {
            volatile tt_l1_ptr uint32_t* s = src + sk_base(e, tb);
            volatile tt_l1_ptr uint32_t* d = dst + row_base(e, tb);
#pragma GCC unroll 8
            for (uint32_t k = 0; k < 8; ++k) {
                d[16 * k] = s[2 * k];
            }
        }
    }
    cb_skout.pop_front(1);
    cb_comb.push_back(1);
}
