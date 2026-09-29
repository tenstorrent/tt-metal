// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// out[r] = sum_{i < num_blocks} src[i * block_stride + r] for r < num_rows (row-major bf16 src [.., rows, H], e.g. a
// gather over num_blocks chips) -> bf16 TILE [1, 1, num_rows, H]: the sum and the tilize in one pass.
struct MoeAgSumRowsTiledParams {
    uint32_t num_rows = 0;
    uint32_t num_blocks = 2;
    uint32_t block_stride = 0;
};
struct MoeAgSumRowsTiledInputs {
    ttnn::Tensor src;
    std::optional<ttnn::Tensor> preallocated_output;
};

// out[i] = a[a_offset + i] + b[b_offset + i] for i < num_rows (row-major bf16, width H) -> row-major [1, 1, num_rows,
// H]; info_offset: b_offset per device from the chip info (word 1: the other row's block start) instead.
struct MoeAgAddRowsParams {
    uint32_t num_rows = 0;
    uint32_t a_offset = 0;
    uint32_t b_offset = 0;
    bool info_offset = false;
    uint32_t batch = 2;  // rows per read barrier
};
struct MoeAgAddRowsInputs {
    ttnn::Tensor a;
    ttnn::Tensor b;
    ttnn::Tensor chip_info;
    std::optional<ttnn::Tensor> preallocated_output;
};

// y bfp8 TILE [.., rows, H] -> row-major bf16 [rows, H], only the tile rows that hold tokens (each local expert's
// ceil(count / 32) tile rows at its region; counts / regions / the local-slot map read on device).
struct MoeAgUntilizeActiveParams {
    uint32_t experts_per_chip = 0;
    uint32_t tiles_per_block = 32;  // W: tiles untilized per block (H % (32 W) == 0)
};
struct MoeAgUntilizeActiveInputs {
    ttnn::Tensor y;
    ttnn::Tensor counts;
    ttnn::Tensor regions;
    ttnn::Tensor local_slot_map;
    std::optional<ttnn::Tensor> preallocated_output;
};

// x bf16 TILE [.., S, H] -> row-major [1, 1, S H / 1024, 1024]: token row g = pages (H / 1024) g .. (H / 1024) g + H /
// 1024 - 1 (2 KB pages: a token's reads spread over H / 1024 DRAM banks).
struct MoeAgUntilizeXParams {};
struct MoeAgUntilizeXInputs {
    ttnn::Tensor x;
    std::optional<ttnn::Tensor> preallocated_output;
};

}  // namespace ttnn::prim
