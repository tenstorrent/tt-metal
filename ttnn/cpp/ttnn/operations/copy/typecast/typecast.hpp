// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

constexpr bool is_8_bit_int(DataType dtype) { return dtype == DataType::UINT8 or dtype == DataType::INT8; }

constexpr bool is_integer_dtype(DataType dtype) {
    return dtype == DataType::INT32 or dtype == DataType::UINT32 or dtype == DataType::UINT16 or
           dtype == DataType::UINT8 or dtype == DataType::INT8;
}

// True when a dtype change converts between an integer and a non-integer (float) format. Data movement ops that
// change dtype only by giving the unpacker and the packer different CB data formats are wrong for these pairs
// (the bits are reinterpreted, not converted); they must go through ttnn::typecast instead.
constexpr bool is_int_float_conversion(DataType from, DataType to) {
    return is_integer_dtype(from) != is_integer_dtype(to);
}

Tensor typecast(
    const Tensor& input,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor typecast(
    const Tensor& input_tensor,
    const DataType& tt_input_dtype,
    const DataType& tt_output_dtype,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn
