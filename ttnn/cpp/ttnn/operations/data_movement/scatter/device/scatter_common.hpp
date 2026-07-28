// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scatter_device_operation_types.hpp"
#include "common/common.hpp"  // Data movement common utilities

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>

namespace ttnn::prim {

// CB index assignments shared by both scatter program factories.
enum class ScatterCB : std::underlying_type_t<tt::CBIndex> {
    INPUT = tt::CBIndex::c_0,
    SRC = tt::CBIndex::c_1,
    INDEX = tt::CBIndex::c_2,
    DST = tt::CBIndex::c_3,
    FP32_TEMP = tt::CBIndex::c_4,
};

constexpr uint32_t BIT_MASK_32 = 32 - 1;

inline uint64_t ceil32(const uint64_t& number) {
    return ((number & BIT_MASK_32) == 0) ? number : ((number | BIT_MASK_32) + 1);
}

// maximal input/index/source/output chunk size, divisible by 32, calculated as follows:
// BH available L1 mem size of nearly 1.5 MB...
// ... divided by 4 to be able to allocate four equally long row chunks (coming from input/index/source/output
// tensors)
// ... divided by 4 to account for 4-byte datum sizes of each tensor (fp32, int32)
// ... minimized by ~10% to account for reserved memory
inline uint32_t calculate_optimal_chunk_size(const ttnn::Tensor& input_tensor) {
    uint32_t l1_per_chunk = (ttnn::operations::data_movement::get_max_l1_space(input_tensor) / 4) / 4;
    return ceil32((l1_per_chunk * 9 / 10) - 32);
}

}  // namespace ttnn::prim
