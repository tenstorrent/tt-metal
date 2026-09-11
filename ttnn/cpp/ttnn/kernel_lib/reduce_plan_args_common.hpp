// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::kernel_lib::reduce_plan_args {

// Host and device deliberately share only these small wire-format descriptions.
// Compute kernels receive one call-count word followed by fixed-size calls in
// execution order. Dataflow kernels separately receive one aggregate auxiliary
// recipe for the entire planning unit. Both payloads may be appended after
// kernel-owned compile-time arguments.
inline constexpr std::uint32_t call_count_word_count = 1;
inline constexpr std::uint32_t auxiliary_header_word_count = 1;
inline constexpr std::uint32_t no_cb_id = 0xFF;
inline constexpr std::uint32_t float_one_bits = 0x3F800000;

// Calls carry an offset and count into the sequence-level auxiliary recipe;
// physical tile descriptions never appear in the compute payload.
enum class CallWord : std::uint32_t {
    Configuration,
    CircularBuffers,
    Rows,
    Columns,
    Batches,
    RowStride,
    ReduceFactor,
    ReduceAxisChunkTiles,
    ChunkAndAuxiliary,
    PostScaleBits,
    AccumulationIndex,
    Count,
};

inline constexpr std::uint32_t call_word_count = static_cast<std::uint32_t>(CallWord::Count);

enum class Math : std::uint32_t { Sum, Average, Maximum, Minimum };
enum class Dimension : std::uint32_t { Row, Column, Scalar };

namespace config {
inline constexpr std::uint32_t path_shift = 0;
inline constexpr std::uint32_t path_mask = 0x1;
inline constexpr std::uint32_t math_shift = 1;
inline constexpr std::uint32_t math_mask = 0x3;
inline constexpr std::uint32_t dimension_shift = 3;
inline constexpr std::uint32_t dimension_mask = 0x3;
inline constexpr std::uint32_t fp32_mode_shift = 5;
inline constexpr std::uint32_t fp32_mode_mask = 0x1;
inline constexpr std::uint32_t algorithm_shift = 6;
inline constexpr std::uint32_t algorithm_mask = 0x1;
inline constexpr std::uint32_t input_policy_shift = 7;
inline constexpr std::uint32_t input_policy_mask = 0x7;
inline constexpr std::uint32_t reload_mode_shift = 10;
inline constexpr std::uint32_t reload_mode_mask = 0x7;
inline constexpr std::uint32_t reconfig_mode_shift = 13;
inline constexpr std::uint32_t reconfig_mode_mask = 0x3;
inline constexpr std::uint32_t within_tile_shift = 15;
inline constexpr std::uint32_t within_tile_mask = 0x1;
inline constexpr std::uint32_t accumulation_mode_shift = 16;
inline constexpr std::uint32_t accumulation_mode_mask = 0x3;
inline constexpr std::uint32_t partial_mode_shift = 18;
inline constexpr std::uint32_t partial_mode_mask = 0x3;
}  // namespace config

namespace circular_buffers {
inline constexpr std::uint32_t input_shift = 0;
inline constexpr std::uint32_t auxiliary_shift = 8;
inline constexpr std::uint32_t output_shift = 16;
inline constexpr std::uint32_t accumulator_shift = 24;
inline constexpr std::uint32_t id_mask = 0xFF;
}  // namespace circular_buffers

namespace chunk_and_auxiliary {
inline constexpr std::uint32_t output_tiles_shift = 0;
inline constexpr std::uint32_t output_tiles_mask = 0xFF;
inline constexpr std::uint32_t auxiliary_tile_offset_shift = 8;
inline constexpr std::uint32_t auxiliary_tile_offset_mask = 0xFF;
inline constexpr std::uint32_t auxiliary_tile_count_shift = 16;
inline constexpr std::uint32_t auxiliary_tile_count_mask = 0xFF;
}  // namespace chunk_and_auxiliary

namespace auxiliary_header {
inline constexpr std::uint32_t cb_id_shift = 0;
inline constexpr std::uint32_t cb_id_mask = 0xFF;
inline constexpr std::uint32_t tile_count_shift = 8;
inline constexpr std::uint32_t tile_count_mask = 0xFF;
}  // namespace auxiliary_header

enum class AuxiliaryTileWord : std::uint32_t {
    Configuration,
    ValueBits,
    Count,
};

inline constexpr std::uint32_t auxiliary_tile_word_count = static_cast<std::uint32_t>(AuxiliaryTileWord::Count);

namespace auxiliary_configuration {
inline constexpr std::uint32_t tile_type_shift = 0;
inline constexpr std::uint32_t tile_type_mask = 0x3;
inline constexpr std::uint32_t valid_elements_shift = 2;
inline constexpr std::uint32_t valid_elements_mask = 0xFFFF;
}  // namespace auxiliary_configuration

constexpr std::uint32_t extract(std::uint32_t word, std::uint32_t shift, std::uint32_t mask) {
    return (word >> shift) & mask;
}

constexpr std::uint32_t insert(std::uint32_t value, std::uint32_t shift, std::uint32_t mask) {
    return (value & mask) << shift;
}

constexpr std::uint32_t call_word_offset(std::uint32_t call_offset, CallWord word) {
    return call_offset + static_cast<std::uint32_t>(word);
}

constexpr std::uint32_t auxiliary_tiles_offset(std::uint32_t auxiliary_offset) {
    return auxiliary_offset + auxiliary_header_word_count;
}

constexpr std::uint32_t auxiliary_tile_offset(std::uint32_t auxiliary_tiles_offset, std::uint32_t tile_index) {
    return auxiliary_tiles_offset + tile_index * auxiliary_tile_word_count;
}

constexpr std::uint32_t auxiliary_tile_word_offset(
    std::uint32_t auxiliary_tiles_offset, std::uint32_t tile_index, AuxiliaryTileWord word) {
    return auxiliary_tile_offset(auxiliary_tiles_offset, tile_index) + static_cast<std::uint32_t>(word);
}

constexpr std::uint32_t call_compile_time_arg_count() { return call_word_count; }

constexpr std::uint32_t auxiliary_compile_time_arg_count(std::uint32_t auxiliary_tile_count) {
    return auxiliary_header_word_count + auxiliary_tile_count * auxiliary_tile_word_count;
}

}  // namespace ttnn::kernel_lib::reduce_plan_args
