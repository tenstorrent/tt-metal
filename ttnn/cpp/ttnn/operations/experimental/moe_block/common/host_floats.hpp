// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host weights as floats (the MoE head's and the MoE tail's weight preparation).

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block {

// A host float32 / bfloat16 tensor's values; op: the diagnostics' prefix ("moe_head").
inline std::vector<float> to_floats(const Tensor& t, std::string_view op) {
    TT_FATAL(t.storage_type() == StorageType::HOST, "{}: the weights come from the host", op);
    if (t.dtype() == tt::tt_metal::DataType::FLOAT32) {
        return t.to_vector<float>();
    }
    TT_FATAL(
        t.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "{}: host weights as float32 or bfloat16, got {}",
        op,
        t.dtype());
    const auto v = t.to_vector<bfloat16>();
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<float>(v[i]);
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block
