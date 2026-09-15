// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

////////////////////////////////////////////////////////////////
// Where our chunks land in the output. This is the concatenation, and it is the whole of what
// all_gather tells the walk in kernels/chunk_walk.hpp.
//
// Glossary:
//   stripe    -- our chunks per output row. For a last-dim gather: last_dim / devices / tile width.
//   row       -- one output row of chunks: every device's stripe, side by side.
//   out_chunk -- one of our chunk ids -> its chunk id in the output.
//   row_room  -- our chunks left in this row. A run stops there: past it sits another device's
//                stripe, which is not ours to write.
//
// Rows are strided. Our stripe repeats every num_devices * stripe chunks in the output, and the
// gap in between holds the other devices' stripes. That stride is why runs are short, and so it
// is also why the host sizes packets and CB entries the way it does (see device/chunk_plan.hpp).
////////////////////////////////////////////////////////////////

// `base` is where our stripe starts in a row: stripe_index * stripe.
template <uint32_t stripe, uint32_t num_devices>
FORCE_INLINE uint32_t out_chunk(uint32_t ours, uint32_t base) {
    const uint32_t row = ours / stripe;
    return base + ours + row * (num_devices - 1) * stripe;
}

template <uint32_t stripe>
FORCE_INLINE uint32_t row_room(uint32_t ours) {
    return stripe - ours % stripe;
}
