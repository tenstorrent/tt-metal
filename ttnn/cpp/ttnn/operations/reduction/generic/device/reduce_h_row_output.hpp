// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the tiled reduce-H factories (reduction/generic and experimental/quasar/reduction).
//
// On Quasar the FPU column reduce (REDUCE_COL) leaves its result in DEST as a 1x32 tile: one 1x16 row per face
// column, MAX_FPU_ROWS rows apart. It must be packed into a 1x32-tile output buffer, then expanded back to the op's
// 32x32 output page by kRowExpandWriter. The SFPU column reduce keeps the full 32x32 layout, so it is packed as before.

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-umd/types/arch.hpp>

#include "ttnn/tensor/types.hpp"

namespace ttnn::prim::reduce_h_row_output {

inline constexpr const char* kRowExpandWriter =
    "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_reduce_h_expand_row_metal2.cpp";

// True when the H reduce's compute packs a 1x32 row per output tile. Mirrors the Quasar branch of
// compute_kernel_lib::is_sfpu_reduce_path: there only bf16 MIN takes the SFPU column reduce.
// `is_min` is ReduceOpMath::MIN; the enum is not named here because the generic and Quasar reduce
// directories each declare their own copy of it.
inline bool packs_1x32_row(tt::ARCH arch, tt::tt_metal::DataType input_dtype, bool is_min) {
    return arch == tt::ARCH::QUASAR && !(input_dtype == tt::tt_metal::DataType::BFLOAT16 && is_min);
}

// kRowExpandWriter copies datums one by one into a 32x32 tile page, so it needs 32x32 tiles and a plain
// (non-block-float) output format.
inline void check_supported(uint32_t tile_height, uint32_t tile_width, tt::DataFormat out_format, const char* op_name) {
    TT_FATAL(
        tile_height == tt::constants::TILE_HEIGHT && tile_width == tt::constants::TILE_WIDTH,
        "{} on Quasar expects 32x32 input tiles, got {}x{}",
        op_name,
        tile_height,
        tile_width);
    TT_FATAL(
        out_format == tt::DataFormat::Float16_b || out_format == tt::DataFormat::Float32 ||
            out_format == tt::DataFormat::Int32 || out_format == tt::DataFormat::UInt32,
        "{} on Quasar supports BFLOAT16, FLOAT32, INT32 and UINT32 output, got {}",
        op_name,
        out_format);
}

}  // namespace ttnn::prim::reduce_h_row_output
