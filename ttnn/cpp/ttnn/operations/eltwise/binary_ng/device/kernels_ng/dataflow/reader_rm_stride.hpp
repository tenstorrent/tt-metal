// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"

// The host passes one byte stride, computed from operand A. A chunk is a run of elements;
// each circular buffer is pitched by that operand's own element size, so a bf16 row and an
// fp32 row of the same width land on different byte offsets.
struct RmOperandChunk {
    uint32_t offset_a;
    uint32_t bytes_a;
    uint32_t read_len_a;
    uint32_t offset_b;
    uint32_t bytes_b;
    uint32_t read_len_b;
    uint32_t elements;
};

FORCE_INLINE RmOperandChunk rm_operand_chunk(
    uint32_t tile_index,
    uint32_t stride_size_bytes_a,
    uint32_t row_width_bytes_a,
    uint32_t element_size_a,
    uint32_t element_size_b,
    uint32_t alignment_a,
    uint32_t alignment_b) {
    const uint32_t offset_a = tile_index * stride_size_bytes_a;
    const uint32_t bytes_left_a = row_width_bytes_a - offset_a;
    const uint32_t bytes_a = (stride_size_bytes_a < bytes_left_a) ? stride_size_bytes_a : bytes_left_a;
    const uint32_t elements = bytes_a / element_size_a;
    const uint32_t element_offset = offset_a / element_size_a;
    const uint32_t offset_b = element_offset * element_size_b;
    const uint32_t bytes_b = elements * element_size_b;
    return RmOperandChunk{
        offset_a,
        bytes_a,
        align(bytes_a, alignment_a),
        offset_b,
        bytes_b,
        align(bytes_b, alignment_b),
        elements,
    };
}
