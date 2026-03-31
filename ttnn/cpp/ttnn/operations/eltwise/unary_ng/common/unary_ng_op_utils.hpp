// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/types.hpp"

#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace ttnn::operations::unary_ng {

/** Compute kernel filename for the given op type (no path prefix). */
std::string_view get_compute_kernel_path(
    unary::UnaryOpType op_type, std::optional<tt::tt_metal::DataType> input_dtype = std::nullopt);

/** Pack a scalar for runtime kernel arg (float / uint32_t / int32_t). */
uint32_t pack_scalar_runtime_arg_impl(float param, tt::tt_metal::DataType dtype);
uint32_t pack_scalar_runtime_arg_impl(std::uint32_t param, tt::tt_metal::DataType dtype);
uint32_t pack_scalar_runtime_arg_impl(std::int32_t param, tt::tt_metal::DataType dtype);

/** Pack scalar from EltwiseUnaryWithParam at given index. */
uint32_t pack_scalar_runtime_arg(const unary::EltwiseUnaryWithParam& op, size_t index, tt::tt_metal::DataType dtype);

/** Whether the op uses approximate math (for unary_ng: false for all). */
bool get_op_approx_mode(unary::UnaryOpType op_type);

/** Block defines for kernel compilation (SFPU_OP_CHAIN_*, include macros). */
std::map<std::string, std::string> get_block_defines(
    const std::vector<unary::EltwiseUnaryWithParam>& op_chain,
    const std::string& block_id = "0",
    const std::string& idst = "0",
    std::optional<tt::tt_metal::DataType> input_dtype = std::nullopt);

}  // namespace ttnn::operations::unary_ng
