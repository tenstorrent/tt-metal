// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

// W global two-pass reorder: within a slice of nf frames × h_total rows, emit all interior rows first
// (nf*input_H_dev) then all corner rows (nf*2*padding_h), so the W exchange does its H-independent
// interior work while H is still producing and only reaches the H→W corner gate once H's per-batch
// corner commit is done (no stall). Returns (frame, h_padded). The W reader and W writer call this in
// lock-step to address the halo buffer identically — keep it shared so the two can never drift.
inline void np_reorder_batch(
    uint32_t k, uint32_t nf, uint32_t input_H_dev, uint32_t padding_h, uint32_t& frame, uint32_t& h_padded) {
    const uint32_t interior = nf * input_H_dev;
    if (k < interior) {
        frame = k / input_H_dev;
        h_padded = padding_h + (k % input_H_dev);
    } else {
        const uint32_t kc = k - interior;
        const uint32_t cpf = 2 * padding_h;  // corners per frame (top + bottom)
        frame = kc / cpf;
        const uint32_t c = kc % cpf;
        h_padded = (c < padding_h) ? c : (padding_h + input_H_dev + (c - padding_h));
    }
}
