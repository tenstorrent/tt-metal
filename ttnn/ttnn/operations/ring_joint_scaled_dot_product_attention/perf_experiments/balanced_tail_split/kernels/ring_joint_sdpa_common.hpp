// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared by the compute-core reader, compute and writer kernels: the attention-unit decode. Units are ordered
// for (b, q head): [main q blocks], [joint q blocks] (head-major); core c owns units c, c + C, c + 2C, ...
// Every unit processes a nominal Sq-row block (CB credits are always nominal, so every block of every CB starts on
// the same slot); `rows` is the number of valid q tile-rows (reads and writes cover only those).
#pragma once

#include <cstdint>

struct AttnUnit {
    uint32_t bh;    // b * hq + h
    uint32_t row0;  // first q tile-row of the block within its tensor
    uint32_t rows;  // valid q tile-rows (<= Sq)
    bool joint;     // joint-Q unit (writes joint_out)
};

template <uint32_t Sq, uint32_t n_q_t, uint32_t Lq_t>
inline AttnUnit decode_unit(uint32_t u) {
    constexpr uint32_t nqb_main = (n_q_t + Sq - 1) / Sq;
    constexpr uint32_t nqb_joint = (Lq_t + Sq - 1) / Sq;
    constexpr uint32_t Uq = nqb_main + nqb_joint;
    AttnUnit a;
    a.bh = u / Uq;
    const uint32_t qb = u - a.bh * Uq;
    if (qb < nqb_main) {
        a.joint = false;
        a.row0 = qb * Sq;
        a.rows = (n_q_t - a.row0) < Sq ? (n_q_t - a.row0) : Sq;
    } else {
        a.joint = true;
        a.row0 = (qb - nqb_main) * Sq;
        a.rows = (Lq_t - a.row0) < Sq ? (Lq_t - a.row0) : Sq;
    }
    return a;
}

template <uint32_t Sq, uint32_t n_q_t, uint32_t Lq_t, uint32_t BH, uint32_t C>
constexpr uint32_t total_units() {
    return BH * ((n_q_t + Sq - 1) / Sq + (Lq_t + Sq - 1) / Sq);
}

inline uint32_t units_of_core(uint32_t U, uint32_t C, uint32_t c) { return c < U ? (U - c + C - 1) / C : 0; }

// Balanced tail split (host: plan_work_split): every core runs W = U / C whole units (c, c + C, ...), then at most
// one TAIL PIECE (the remainder units re-blocked into r-row pieces, r | Sq) read from 5 runtime args at `a`:
// has_tail, bh, row0, rows, joint. The piece is always the core's last unit.
inline AttnUnit tail_piece(uint32_t a) {
    AttnUnit t;
    t.bh = get_arg_val<uint32_t>(a + 1);
    t.row0 = get_arg_val<uint32_t>(a + 2);
    t.rows = get_arg_val<uint32_t>(a + 3);
    t.joint = get_arg_val<uint32_t>(a + 4) != 0;
    return t;
}
