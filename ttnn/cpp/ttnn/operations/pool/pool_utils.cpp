// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::pool {

std::map<std::string, std::string> get_defines(Pool2DType pool_type) {
    std::map<std::string, std::string> defines;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: defines["REDUCE_OP"] = "PoolType::MAX"; break;
        case Pool2DType::AVG_POOL2D: defines["REDUCE_OP"] = "PoolType::AVG"; break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";

    return defines;
}

uint32_t get_aligned_stick_size(const ttnn::Shape& shape, const Tensor& tensor) {
    const uint32_t stick_nbytes = shape[-1] * tensor.element_size();
    const uint32_t alignment = tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                   ? tt::tt_metal::hal::get_dram_alignment()
                                   : tt::tt_metal::hal::get_l1_alignment();
    return tt::round_up(stick_nbytes, alignment);
}

}  // namespace ttnn::operations::pool
