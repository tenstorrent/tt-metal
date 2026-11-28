// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_unsqueeze data-movement kernel instance: tile half 0 (input face 0 row 0 -> output face 0 column 0); reader-config
// RISC (NCRISC, NoC0). Both instances run the same face-row move (g6_unsqueeze_face_move.hpp); the half each one owns
// comes from CT arg half_begin. With HALVES_PER_RISC = 2 only the reader instance is launched and
// moves both halves. There is no compute kernel (pure bit move; any FPU path narrows fp32).

#include "g6_unsqueeze_face_move.hpp"

void kernel_main() { g6_unsqueeze::kernel_body(); }
