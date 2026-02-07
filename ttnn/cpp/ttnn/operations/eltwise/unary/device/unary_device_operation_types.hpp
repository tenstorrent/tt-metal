// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include <tuple>

namespace ttnn::prim {

using ttnn::operations::unary::EltwiseUnaryWithParam;

struct UnaryParams {
    const std::vector<EltwiseUnaryWithParam> op_chain;
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const bool fp32_dest_acc_en = false;
    const bool preserve_fp32_precision = false;
    const bool bfp8_pack_precise = false;
    const std::optional<CoreRangeSet> sub_core_grids = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple("op_chain", "output_dtype", "output_memory_config", "fp32_dest_acc_en", "preserve_fp32_precision", "bfp8_pack_precise", "sub_core_grids");
    auto attribute_values() const { return std::forward_as_tuple(op_chain, output_dtype, output_memory_config, fp32_dest_acc_en, preserve_fp32_precision, bfp8_pack_precise, sub_core_grids); }
};

struct UnaryInputs {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input, preallocated_output); }
};

}  // namespace ttnn::prim
