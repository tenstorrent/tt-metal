// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::topk_large_indices::program {

// Shared host/device encoding for the compile-time row-reduction body.
enum class ComputeBodyMode : uint32_t {
    Classic = 0,
    FusedEndToEnd = 1,
    FusedSegmented = 2,
};

// Shared host/device encoding for the compile-time sort-key mode of FP32 input.
//   None:    BF16 input, the fused word is [bf16 value | u16 index].
//   Tf32:    FP32 input unpacked through SrcA, so DEST holds TF32 (low 13 bits zero) and a <= 2-bit
//            chunk field fits under it; the key keeps all 10 TF32 mantissa bits.
//   Shifted: FP32 input unpacked to DEST at full precision; the key is trunc(x - T) for a per-row
//            estimate T of the k-th score, above a chunk field of up to 5 bits.
enum class Fp32Key : uint32_t {
    None = 0,
    Tf32 = 1,
    Shifted = 2,
};

}  // namespace ttnn::operations::experimental::topk_large_indices::program
