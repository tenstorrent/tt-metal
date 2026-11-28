// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <variant>

#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

// Python-facing pad value: an exact integer or a float. Python ints are taken exactly (nanobind tries the int64_t
// alternative first), so int32 and uint32 values that float cannot represent survive the binding.
using PadValueArg = std::variant<int64_t, float>;

// Maps a binding value onto the PadValue the ops take. A non-negative integer in uint32 range uses the uint32_t arm.
// A negative integer in int32 range on an INT32 tensor uses the two's-complement word in the uint32_t arm. Anything
// else falls back to the float arm.
inline PadValue to_pad_value(const PadValueArg& arg, DataType dtype) {
    if (const auto* integer = std::get_if<int64_t>(&arg)) {
        const int64_t v = *integer;
        if (v >= 0 && v <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            return PadValue{static_cast<uint32_t>(v)};
        }
        if (dtype == DataType::INT32 && v < 0 && v >= static_cast<int64_t>(std::numeric_limits<int32_t>::min())) {
            return PadValue{std::bit_cast<uint32_t>(static_cast<int32_t>(v))};
        }
        return PadValue{static_cast<float>(v)};
    }
    return PadValue{std::get<float>(arg)};
}

inline std::optional<PadValue> to_pad_value(const std::optional<PadValueArg>& arg, DataType dtype) {
    if (!arg.has_value()) {
        return std::nullopt;
    }
    return to_pad_value(*arg, dtype);
}

}  // namespace ttnn::operations::data_movement
