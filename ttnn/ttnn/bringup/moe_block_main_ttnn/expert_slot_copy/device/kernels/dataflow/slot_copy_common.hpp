// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert_slot_copy's shared kernel layout (see ../../expert_slot_copy_device_operation_types.hpp for the host side).
//
// Runtime args, the same layout on the reader and the writer of a port core (the writer appends its fabric
// connection's):
//   0..5   src_gu, src_d, dst_gu, dst_d, runs, sems  (buffer addresses)
//   6      this chip's index r C + c
//   7      jobs on this core (<= ESC_MAXJ)
//   8 ..   ESC_MAXJ jobs of 3 words: w0 = table | dst << 8 | part << 16 | parts << 24 (dst ESC_LOCAL: a local copy),
//          w1 = the destination's fabric chip id | mesh id << 16, w2 = its receiver core x | y << 16 (NoC coordinates)
//   RCV0   readies this core sends (0: not a receiver here); then the source's fabric chip id | mesh id << 16; then
//          ESC_MAXL sender cores (x | y << 16) on the source chip
// A run table (one page of `runs`): word 0 the runs n, then n runs of ESC_RUN_WORDS words: weight (0 gate/up, 1 down),
// source bank, source byte offset, destination bank, destination byte offset, bytes. A table's chunks (each run cut in
// pieces of ESC_CHUNK bytes, in order) are numbered across its runs; chunk g belongs to part g % parts.
//
// Semaphore words (sems, 16 B apart, the same address on every port core of every chip): word 0 the done count of
// the receiver, word 1 + j the ready word of destination chip j at a sender.

#pragma once

#include <cstdint>

namespace esc {

constexpr uint32_t MAXJ = ESC_MAXJ;
constexpr uint32_t MAXL = ESC_MAXL;
constexpr uint32_t LOCAL = 0xFF;
constexpr uint32_t RUN_WORDS = 6;
constexpr uint32_t SEM_STRIDE = 16;
constexpr uint32_t ARG_JOBS = 8;
constexpr uint32_t RCV0 = ARG_JOBS + 3 * MAXJ;
constexpr uint32_t NARGS = RCV0 + 2 + MAXL;  // (the writer's fabric connection args follow)

struct Job {
    uint32_t table, dst, part, parts;
    uint32_t fabric_chip, fabric_mesh;
    uint32_t rx, ry;
};

FORCE_INLINE Job job(uint32_t j) {
    const uint32_t w0 = get_arg_val<uint32_t>(ARG_JOBS + 3 * j);
    const uint32_t w1 = get_arg_val<uint32_t>(ARG_JOBS + 3 * j + 1);
    const uint32_t w2 = get_arg_val<uint32_t>(ARG_JOBS + 3 * j + 2);
    return Job{
        w0 & 0xFF,
        (w0 >> 8) & 0xFF,
        (w0 >> 16) & 0xFF,
        (w0 >> 24) & 0xFF,
        w1 & 0xFFFF,
        w1 >> 16,
        w2 & 0xFFFF,
        w2 >> 16};
}

// Calls f(weight, src_bank, src_off, dst_bank, dst_off, bytes) for every chunk of the table at tab (L1) that is this
// part's.
template <typename F>
FORCE_INLINE void for_my_chunks(const volatile tt_l1_ptr uint32_t* tab, uint32_t part, uint32_t parts, F&& f) {
    const uint32_t n = tab[0];
    uint32_t g = 0;
    for (uint32_t r = 0; r < n; ++r) {
        const volatile tt_l1_ptr uint32_t* e = tab + 1 + r * RUN_WORDS;
        const uint32_t w = e[0], sb = e[1], so = e[2], db = e[3], dof = e[4], bytes = e[5];
        for (uint32_t o = 0; o < bytes; o += ESC_CHUNK, ++g) {
            if (g % parts == part) {
                const uint32_t len = bytes - o < ESC_CHUNK ? bytes - o : ESC_CHUNK;
                f(w, sb, so + o, db, dof + o, len);
            }
        }
    }
}

}  // namespace esc
