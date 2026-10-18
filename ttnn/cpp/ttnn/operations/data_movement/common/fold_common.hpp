// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <fmt/core.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/tensor/tensor.hpp"

// Single source of truth for the fold tile-native gate, shared by data_movement::Fold (WH/BH)
// and experimental::quasar::Fold to prevent divergence between the two routing predicates.
namespace ttnn::operations::data_movement::fold_common {

// > 1 lets untilize fill one group while the writer drains the previous; factory + predicate scale by this.
inline constexpr uint32_t kFoldSrcCbDepthPerCTile = 2;

// hal budget assumes ringbuf=0 but DFB allocs start past KERNEL_CONFIG (~105 KB WH); reserve
// covers that gap + JIT code/stack so the predicate stays a bound.
inline constexpr uint64_t kFoldL1CodeStackReserveBytes = 160 * 1024;

// FLOAT32/UINT16 pass through; every other input dtype collapses to BFLOAT16 on RM output.
inline tt::tt_metal::DataType fold_output_dtype(tt::tt_metal::DataType input_dtype) {
    return (input_dtype == tt::tt_metal::DataType::FLOAT32 || input_dtype == tt::tt_metal::DataType::UINT16)
               ? input_dtype
               : tt::tt_metal::DataType::BFLOAT16;
}

// Bytes the writer's per-super-block RM scratch needs (one output row of contiguous sticks).
inline uint64_t tile_native_fold_scratch_bytes(const Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w) {
    const uint32_t out_elem = tt::datum_size(datatype_to_dataformat_converter(fold_output_dtype(input_tensor.dtype())));
    const auto& shape = input_tensor.logical_shape();
    const uint32_t input_width = shape[2];
    const uint32_t C = shape[-1];
    return static_cast<uint64_t>(input_width / stride_w) * stride_h * stride_w * C * out_elem;
}

// nullopt = supported, else a short reason ("stride_h=0 …", "sharded input", "c_bytes=6 (not 16B-aligned)", …).
inline std::optional<std::string> tile_native_fold_rejection_reason(
    const Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w) {
    // Composite consults this before the primitive; unguarded input_width/stride_w SIGFPEs the host.
    if (stride_h == 0 || stride_w == 0) {
        return fmt::format("stride_h={} or stride_w={} (must be > 0)", stride_h, stride_w);
    }
    if (input_tensor.layout() != tt::tt_metal::Layout::TILE) {
        return std::string{"non-TILE layout"};
    }
    if (input_tensor.is_sharded()) {
        return std::string{"sharded input"};
    }
    // tt_memmove hits the NoC self-copy path only with 16B-aligned c_bytes; else per-pixel CPU memmove (~2x slower).
    const uint32_t out_elem = tt::datum_size(datatype_to_dataformat_converter(fold_output_dtype(input_tensor.dtype())));
    const uint32_t c_bytes = input_tensor.logical_shape()[-1] * out_elem;
    if (c_bytes % 16 != 0) {
        return fmt::format("c_bytes={} (not 16B-aligned)", c_bytes);
    }
    const uint64_t scratch = tile_native_fold_scratch_bytes(input_tensor, stride_h, stride_w);
    const auto in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto out_df = datatype_to_dataformat_converter(fold_output_dtype(input_tensor.dtype()));
    const uint32_t c_tiles = tt::div_up(input_tensor.padded_shape()[-1], tt::constants::TILE_WIDTH);
    // Matches the factory: SRC0+SRC1 each carry kFoldSrcCbDepthPerCTile × c_tiles so routing tracks alloc.
    const uint64_t cb_bytes =
        static_cast<uint64_t>(tt::tile_size(in_df) + tt::tile_size(out_df)) * c_tiles * kFoldSrcCbDepthPerCTile;
    // Static budget keeps routing a pure function of inputs; real fragmentation still surfaces at CB alloc.
    const uint64_t budget = tt::tt_metal::hal::get_max_worker_l1_unreserved_size();
    if (scratch + cb_bytes + kFoldL1CodeStackReserveBytes >= budget) {
        return fmt::format("scratch={} B + CBs={} B exceed L1 budget ({} B)", scratch, cb_bytes, budget);
    }
    return std::nullopt;
}

inline bool is_tile_native_fold_supported(const Tensor& t, uint32_t stride_h, uint32_t stride_w) {
    return !tile_native_fold_rejection_reason(t, stride_h, stride_w).has_value();
}

}  // namespace ttnn::operations::data_movement::fold_common
