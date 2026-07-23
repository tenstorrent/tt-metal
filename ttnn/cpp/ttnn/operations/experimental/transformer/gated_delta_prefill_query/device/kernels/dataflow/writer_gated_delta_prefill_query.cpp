// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Generator kernel for the gated-delta prefill-then-query op.
//
// Builds two constant 32x32 bf16 tiles in L1 (via the gated_delta_mask_gen.hpp helpers, which
// handle the tiled face layout):
//   * cb_mask  : strict-lower mask — 1.0 strictly below the diagonal, 0 on and above it.
//   * cb_ident : identity matrix   — 1.0 on the diagonal, 0 elsewhere.
// The compute kernel uses these to turn each K @ K^T tile into unit lower-triangular form
// (out = gram (*) mask + ident).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

#include "gated_delta_mask_gen.hpp"

void kernel_main() {
    constexpr uint32_t cb_mask = tt::CBIndex::c_2;
    constexpr uint32_t cb_ident = tt::CBIndex::c_3;

    CircularBuffer cb_mask_o(cb_mask);
    CircularBuffer cb_ident_o(cb_ident);

    cb_mask_o.reserve_back(1);
    cb_ident_o.reserve_back(1);
    gated_delta::generate_strict_lower_mask(cb_mask_o.get_write_ptr());
    gated_delta::generate_identity(cb_ident_o.get_write_ptr());
    cb_mask_o.push_back(1);
    cb_ident_o.push_back(1);
}
