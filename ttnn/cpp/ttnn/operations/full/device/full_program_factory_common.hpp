// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

// Convert by dtype, not by which variant the caller passed: fill_value is "float or int" for any dtype.
inline fill_value_t encode_fill_value(const std::variant<float, int>& fill_value, tt::tt_metal::DataType dtype) {
    auto as_float = [&] {
        return std::holds_alternative<float>(fill_value) ? std::get<float>(fill_value)
                                                         : static_cast<float>(std::get<int>(fill_value));
    };

    fill_value_t u;
    switch (dtype) {
        case tt::tt_metal::DataType::INT32: {
            int value = std::holds_alternative<int>(fill_value) ? std::get<int>(fill_value)
                                                                : static_cast<int>(std::get<float>(fill_value));
            u.u32 = static_cast<uint32_t>(value);
            break;
        }
        case tt::tt_metal::DataType::BFLOAT16:
            u.u32 = static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(as_float()))) << 16;
            break;
        default:  // FLOAT32
            u.f32 = as_float();
            break;
    }
    return u;
}

inline std::map<std::string, std::string> get_writer_defines(tt::tt_metal::DataType dtype) {
    std::map<std::string, std::string> defines;
    switch (dtype) {
        case tt::tt_metal::DataType::BFLOAT16: defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case tt::tt_metal::DataType::INT32: defines["OUTPUT_DTYPE_INT32"] = "1"; break;
        case tt::tt_metal::DataType::FLOAT32: defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }
    return defines;
}

}  // namespace ttnn::operations::full
