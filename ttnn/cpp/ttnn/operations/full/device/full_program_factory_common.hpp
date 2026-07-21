// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <ttnn/tensor/types.hpp>

namespace ttnn::operations::full {

union fill_value_t {
    uint32_t u32;
    float f32;
};

inline fill_value_t encode_fill_value(const std::variant<float, int>& fill_value, tt::tt_metal::DataType dtype) {
    fill_value_t u;
    if (std::holds_alternative<int>(fill_value)) {
        u.u32 = std::get<int>(fill_value);
    } else {
        auto float_val = std::get<float>(fill_value);
        if (dtype == tt::tt_metal::DataType::BFLOAT16) {
            u.u32 = static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(float_val))) << 16;
        } else {
            u.f32 = float_val;
        }
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

inline tt::tt_metal::KernelDescriptor::Defines defines_from_map(const std::map<std::string, std::string>& m) {
    tt::tt_metal::KernelDescriptor::Defines out;
    out.reserve(m.size());
    for (const auto& [k, v] : m) {
        out.emplace_back(k, v);
    }
    return out;
}

}  // namespace ttnn::operations::full
