// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>

#include <tt-metalium/bfloat16.hpp>
#include <ttnn/tensor/types.hpp>

namespace ttnn::operations::full {

union fill_value_t {
    uint32_t u32;
    float f32;
};

inline fill_value_t encode_fill_value(const std::variant<float, int>& fill_value, DataType dtype) {
    fill_value_t u;
    if (std::holds_alternative<int>(fill_value)) {
        u.u32 = std::get<int>(fill_value);
    } else {
        auto float_val = std::get<float>(fill_value);
        if (dtype == DataType::BFLOAT16) {
            u.u32 = static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(float_val))) << 16;
        } else {
            u.f32 = float_val;
        }
    }
    return u;
}

inline std::map<std::string, std::string> get_writer_defines(DataType dtype) {
    std::map<std::string, std::string> defines;
    switch (dtype) {
        case DataType::BFLOAT16: defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::INT32: defines["OUTPUT_DTYPE_INT32"] = "1"; break;
        case DataType::FLOAT32: defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }
    return defines;
}

}  // namespace ttnn::operations::full
