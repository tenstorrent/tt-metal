// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::pool {

// Stick size in bytes, rounded up to the buffer's DRAM / L1 alignment.
inline uint32_t aligned_stick_size(const ttnn::Shape& shape, const ttnn::Tensor& tensor) {
    const uint32_t stick_nbytes = shape[-1] * tensor.element_size();
    const uint32_t alignment = tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                   ? tt::tt_metal::hal::get_dram_alignment()
                                   : tt::tt_metal::hal::get_l1_alignment();
    return tt::round_up(stick_nbytes, alignment);
}

}  // namespace ttnn::operations::pool
