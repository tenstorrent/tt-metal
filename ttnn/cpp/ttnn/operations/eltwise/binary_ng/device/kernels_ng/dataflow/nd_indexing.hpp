// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// nd_broadcast_factor is computed on host as 1 when input and output collapsed nD
// volumes match, otherwise output_nd_volume / input_nd_volume. That ratio is the
// dim -6 broadcast extent (outer dims are required to match).
FORCE_INLINE uint32_t get_input_nd_index(const uint32_t output_nd_index, const uint32_t nd_broadcast_factor) {
    return nd_broadcast_factor == 1 ? output_nd_index : output_nd_index / nd_broadcast_factor;
}

// output_nd increases by 1, so the mapped input index either stays or advances by 1.
FORCE_INLINE uint32_t nd_loop_shift(
    const uint32_t current_input_nd,
    const uint32_t next_input_nd,
    const uint32_t advance_shift,
    const uint32_t repeat_shift) {
    return next_input_nd == current_input_nd ? repeat_shift : advance_shift;
}

FORCE_INLINE uint32_t advance_nd_ptr(
    const uint32_t ptr, const uint32_t current_input_nd, const uint32_t next_input_nd, const uint32_t stride) {
    return next_input_nd == current_input_nd ? ptr : ptr + stride;
}
