// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Shared by the binary_ng and ternary reader kernels; keep the two in sync by including this header
// rather than copying it.

// nd_broadcast_factor is computed on host as 1 when input and output collapsed nD
// volumes match, otherwise output_nd_volume / input_nd_volume. That ratio is the
// dim -6 broadcast extent (outer dims are required to match).
FORCE_INLINE uint32_t get_input_nd_index(const uint32_t output_nd_index, const uint32_t nd_broadcast_factor) {
    return nd_broadcast_factor == 1 ? output_nd_index : output_nd_index / nd_broadcast_factor;
}

// For readers that keep the nD slice base in a separate pointer: output_nd increases by 1, so the
// mapped input index either stays (dim -6 broadcast, re-read the same slice) or advances by 1.
FORCE_INLINE uint32_t advance_nd_ptr(
    const uint32_t ptr, const uint32_t current_input_nd, const uint32_t next_input_nd, const uint32_t stride) {
    return next_input_nd == current_input_nd ? ptr : ptr + stride;
}

// For readers that carry a single running tile offset through the d/n/c/th loops. By the end of one
// output nD iteration that offset has walked d_span = d_stride * D past the slice it started on,
// whichever broadcast flags are set, so rewind by d_span to get back to the slice base and step
// forward by nD_stride only when the mapped input nD index actually advances.
FORCE_INLINE uint32_t advance_nd_offset(
    const uint32_t tile_offset,
    const uint32_t current_input_nd,
    const uint32_t next_input_nd,
    const uint32_t nD_stride,
    const uint32_t d_span) {
    const uint32_t slice_base = tile_offset - d_span;
    return next_input_nd == current_input_nd ? slice_base : slice_base + nD_stride;
}
