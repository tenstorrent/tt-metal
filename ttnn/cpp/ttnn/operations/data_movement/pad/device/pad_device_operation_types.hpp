// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

struct PadParams {
    ttnn::Shape output_logical_shape;
    ttnn::Shape output_padded_shape;
    ttnn::Shape input_tensor_start;
    ttnn::PadValue pad_value{};
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_multicore{};
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
};

struct PadInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

namespace detail {

// PadValue's uint32_t arm carries an exact integer (a raw two's-complement word for int32); the float arm
// carries a numeric value. These flatten the variant for the pad program factories.
inline uint32_t pad_value_as_u32(const ttnn::PadValue& pad_value) {
    if (std::holds_alternative<uint32_t>(pad_value)) {
        return std::get<uint32_t>(pad_value);
    }
    return static_cast<uint32_t>(std::get<float>(pad_value));
}

inline float pad_value_as_float(const ttnn::PadValue& pad_value) {
    if (std::holds_alternative<float>(pad_value)) {
        return std::get<float>(pad_value);
    }
    return static_cast<float>(static_cast<int32_t>(std::get<uint32_t>(pad_value)));
}

}  // namespace detail

}  // namespace ttnn::prim
