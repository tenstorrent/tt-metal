// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::step_context {

// Words per integer term (see step_context.hpp).
inline constexpr uint32_t kTermWords = 7;
// Rows of a RoPE table's low half: offsets -1 .. 1023 within a 1024-position block.
inline constexpr uint32_t kRopeLoRows = 1025;
inline constexpr uint32_t kRopeBlock = 1024;
// Each integer output is staged in one 64 B slot.
inline constexpr uint32_t kMaxIntElems = 16;
// Integer elements one core evaluates (7 runtime args each); more spread over further cores.
inline constexpr uint32_t kIntElemsPerCore = 32;
// Rows of one RoPE output (one core each, gathered into row 0's core).
inline constexpr uint32_t kMaxRopeRows = 8;

struct StepContextParams {
    // One entry per integer output: its logical shape, dtype (INT32 / UINT32) and kTermWords words per element.
    std::vector<std::vector<uint32_t>> int_shapes;
    std::vector<DataType> int_dtypes;
    std::vector<std::vector<int32_t>> int_terms;
    // One entry per RoPE output triple (cos, sin, -sin): its table, and the packet field + offset of its position.
    std::vector<uint32_t> rope_tables;
    std::vector<uint32_t> rope_fields;
    std::vector<int32_t> rope_offsets;
    // Per RoPE output (empty: all 1 / 32): its rows (row j at position pkt[field + j] + offset) and its output tile
    // height (rows <= height; [1, 1, rows, Rd] TILE tiled (height, 32)).
    std::vector<uint32_t> rope_rows;
    std::vector<uint32_t> rope_tile_heights;
    MemoryConfig output_mem_config;
};

struct StepContextInputs {
    const Tensor& packet;
    std::vector<Tensor> rope_tables;
};

inline uint32_t rope_rows_of(const StepContextParams& p, size_t j) { return p.rope_rows.empty() ? 1u : p.rope_rows[j]; }
inline uint32_t rope_tile_height_of(const StepContextParams& p, size_t j) {
    return p.rope_tile_heights.empty() ? 32u : p.rope_tile_heights[j];
}

using StepContextSpecReturn = std::vector<tt::tt_metal::TensorSpec>;
using StepContextTensorReturn = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::deepseek::step_context
