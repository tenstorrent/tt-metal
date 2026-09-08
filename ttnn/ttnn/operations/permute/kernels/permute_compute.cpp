// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// permute compute — INTENTIONALLY EMPTY IN PHASE 0.
//
// The `whole_tile_relocation` regime is pure data movement: every 32x32 tile is
// relocated intact, so there is no compute phase and this kernel is NOT part of
// the dispatched program (see permute_program_descriptor.py). It is the seat for
// the deferred R3 `within_tile_transpose` regime, which inserts
// transpose_init/transpose_tile between the two existing CBs.

#include "compute_kernel_api/common.h"

void kernel_main() {}
