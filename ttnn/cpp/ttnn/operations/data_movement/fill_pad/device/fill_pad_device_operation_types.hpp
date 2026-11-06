// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <bit>
#include <cstdlib>
#include <cstring>

namespace ttnn::prim {

struct FillPadParams {
    ttnn::PadValue fill_value;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct FillPadInputs {
    Tensor input;
};

// Experimental specialization; all other calls retain the original path.
inline bool gemma_runtime_hmod_enabled(const FillPadParams& attrs, const FillPadInputs& args) {
    const char* flag = std::getenv("GEMMA_EXPERIMENT_FILLPAD_RUNTIME_HMOD");
    if (!flag || std::strcmp(flag, "1") != 0) {
        return false;
    }
    const auto& x = args.input;
    const auto& s = x.logical_shape();
    const auto& p = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return x.storage_type() == ttnn::StorageType::DEVICE && x.dtype() == tt::tt_metal::DataType::BFLOAT16 &&
           x.layout() == tt::tt_metal::Layout::TILE && s.rank() == 3 && p.rank() == 3 && s[0] == 1 && p[0] == 1 &&
           (s[2] == 1344 || s[2] == 5376) && p[2] == s[2] && s[1] > 0 && s[1] <= 8319 && s[1] % 32 != 0 &&
           p[1] == ((s[1] + 31) / 32) * 32 && tile.get_height() == 32 && tile.get_width() == 32 &&
           x.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
           x.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
           attrs.output_mem_config == x.memory_config() && std::holds_alternative<float>(attrs.fill_value) &&
           std::bit_cast<uint32_t>(std::get<float>(attrs.fill_value)) == 0u;
}
}  // namespace ttnn::prim
