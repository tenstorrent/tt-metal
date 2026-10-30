// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The prep compute's constants, built in L1 (not read from DRAM: every producer would fetch the same few pages from
// one bank at kickoff): eye and tril (diagonal included) [Ct x Ct tiles], ones [Ct x Ct tiles], the WY-inverse
// quadrant masks Qtl, Qbr, Q10 (tiles 0, 1, 2 of the u CB). An fp32 tile is four 16x16 faces: (0) rows 0-15 cols 0-15,
// (1) rows 0-15 cols 16-31, (2) rows 16-31 cols 0-15, (3) rows 16-31 cols 16-31.
//
// Built by the phased prep reader, and by the fused PRODUCER WRITER (writer_chunk_gdn_fused.cpp): the writer has
// nothing to do until its first item is computed, so the build runs there in parallel with the reader's first DRAM
// reads instead of in front of them (or inside the reader's read barrier).
//
// issue_eye() / issue_rest(): one face of ones by RISC stores, then NoC loopback reads only: zeros (from MEM_ZEROS)
// into every face that is not a copy of the ones face, and the copies. The two sets of faces are disjoint, so the reads
// need no order among themselves and ride the caller's read barrier together. After that barrier, finish_eye() stores
// the eye diagonal (all the q/k norm needs) and publishes eye; finish_rest() stores the tril pattern and publishes
// tril, ones and mask (the compute waits for those at pp_decay, their first reader).

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "chunk_gdn_handoff.hpp"

namespace gdn_const {

constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7;
// Three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10) packed into one [1,1,32,96] tensor, in the cb_u slot (17).
constexpr uint32_t cb_mask = 17;
static_assert(cb_mask == gdn_handoff::kCbU, "mask CB index drifted from chunk_gdn_handoff.hpp");

constexpr uint32_t kFaceWords = 256, kFaceBytes = kFaceWords * 4;
constexpr uint32_t kOne = 0x3F800000u;

// The two NoC helpers sit out of line because the prep program's binaries run close to the kernel-config budget in
// watcher builds.
// Zeroes `bytes` at byte offset `offset` of cb's write slot by loopback reads of MEM_ZEROS; no barrier.
__attribute__((noinline)) static void zero_cb(
    const Noc& noc, const CircularBuffer& cb, uint32_t bytes, uint32_t offset) {
    noc.async_write_zeros(cb, bytes, {.offset_bytes = offset});
}
// Loopback copy of the face at src_addr (this core's L1) to byte offset `offset` of cb's write slot.
__attribute__((noinline)) static void copy_face(
    const Noc& noc, uint32_t src_addr, const CircularBuffer& cb, uint32_t offset) {
    UnicastEndpoint self;
    const auto src = noc_traits_t<UnicastEndpoint>::src_args_type{
        .noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = src_addr};
    noc.async_read(self, cb, kFaceBytes, src, {.offset_bytes = offset});
}

// The eye: all zero (its diagonal is stored by finish_eye after the barrier).
template <uint32_t Ct>
__attribute__((noinline)) void issue_eye(const Noc& noc) {
    CircularBuffer eye(cb_eye);
    eye.reserve_back(Ct * Ct);
    zero_cb(noc, eye, 4 * Ct * Ct * kFaceBytes, 0);
}

// tril, ones and the masks.
template <uint32_t Ct>
__attribute__((noinline)) void issue_rest(const Noc& noc) {
    constexpr uint32_t cc = Ct * Ct;
    const uint32_t tb_f = get_tile_size(cb_eye);
    CircularBuffer tril(cb_tril), ones(cb_ones), mask(cb_mask);
    tril.reserve_back(cc);
    ones.reserve_back(cc);
    mask.reserve_back(3);
    const uint32_t ones_face = ones.get_write_ptr();
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_face);
    for (uint32_t i = 0; i < kFaceWords; i++) {
        w[i] = kOne;
    }
    asm volatile("fence");
    auto copy = [&](const CircularBuffer& cb, uint32_t tile, uint32_t face) {
        copy_face(noc, ones_face, cb, tile * tb_f + face * kFaceBytes);
    };
    auto zero = [&](const CircularBuffer& cb, uint32_t tile, uint32_t face, uint32_t faces) {
        zero_cb(noc, cb, faces * kFaceBytes, tile * tb_f + face * kFaceBytes);
    };
    for (uint32_t t = 0; t < cc; t++) {
        for (uint32_t f = (t == 0) ? 1u : 0u; f < 4; f++) {
            copy(ones, t, f);
        }
    }
    for (uint32_t rt = 0; rt < Ct; rt++) {
        for (uint32_t ct = 0; ct < Ct; ct++) {
            const uint32_t t = rt * Ct + ct;
            if (ct < rt) {
                for (uint32_t f = 0; f < 4; f++) {
                    copy(tril, t, f);
                }
            } else if (ct > rt) {
                zero(tril, t, 0, 4);
            } else {
                zero(tril, t, 0, 2);  // faces 0 (pattern, stored after the barrier) and 1
                copy(tril, t, 2);
                zero(tril, t, 3, 1);  // face 3 (pattern)
            }
        }
    }
    constexpr uint32_t kMaskFaces[3] = {0, 3, 2};  // Qtl, Qbr, Q10: the all-ones quadrant of each mask tile
    for (uint32_t t = 0; t < 3; t++) {
        for (uint32_t f = 0; f < 4; f++) {
            if (f == kMaskFaces[t]) {
                copy(mask, t, f);
            } else {
                zero(mask, t, f, 1);
            }
        }
    }
}

// The eye diagonal (32 stores per diagonal tile), then publish eye.
template <uint32_t Ct>
__attribute__((noinline)) void finish_eye() {
    const uint32_t tb_f = get_tile_size(cb_eye);
    CircularBuffer eye(cb_eye);
    for (uint32_t t = 0; t < Ct; t++) {
        volatile tt_l1_ptr uint32_t* e =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eye.get_write_ptr() + (t * Ct + t) * tb_f);
        for (uint32_t f = 0; f < 4 * kFaceWords; f += 3 * kFaceWords) {  // faces 0 and 3
            for (uint32_t r = 0; r < 16; r++) {
                e[f + r * 17] = kOne;
            }
        }
    }
    asm volatile("fence");
    eye.push_back(Ct * Ct);
}

// The tril pattern (272 stores per diagonal tile), then publish tril, ones and the masks.
template <uint32_t Ct>
__attribute__((noinline)) void finish_rest() {
    const uint32_t tb_f = get_tile_size(cb_eye);
    CircularBuffer tril(cb_tril), ones(cb_ones), mask(cb_mask);
    for (uint32_t t = 0; t < Ct; t++) {
        volatile tt_l1_ptr uint32_t* l =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tril.get_write_ptr() + (t * Ct + t) * tb_f);
        for (uint32_t f = 0; f < 4 * kFaceWords; f += 3 * kFaceWords) {  // faces 0 and 3
            for (uint32_t r = 0; r < 16; r++) {
                for (uint32_t q = 0; q <= r; q++) {
                    l[f + r * 16 + q] = kOne;
                }
            }
        }
    }
    asm volatile("fence");
    tril.push_back(Ct * Ct);
    ones.push_back(Ct * Ct);
    mask.push_back(3);
}

}  // namespace gdn_const
