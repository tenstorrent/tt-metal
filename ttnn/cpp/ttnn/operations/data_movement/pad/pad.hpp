// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include <ranges>
#include <string_view>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::data_movement {

struct PadSpecDim {
    uint32_t before_elements;
    uint32_t after_elements;
};

}  // namespace operations::data_movement

// This function signature is similar to pytorch's signature
// Any rank tensor supported
//
// use_multicore defaults to true here to match the Python binding's default
// (pad_nanobind.cpp). Aligned defaults prevent callers from silently routing
// to a different code path than Python-driven tests cover.
//
// implementation selects "auto" (default; codegen when in-scope and not perf-demoted, else
// native), "native" (always the existing device_operation prim), or "codegen" (always
// prim::pad_codegen; TT_FATALs if the input is out of the codegen path's scope).
ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<operations::data_movement::PadSpecDim>& padding,
    float value,
    bool use_multicore = true,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    std::string_view implementation = "auto");

ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<std::array<uint32_t, 2>>& padding,
    float value,
    bool use_multicore = true,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    std::string_view implementation = "auto");

// legacy API
ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::Array4D& output_padded_shape,
    const tt::tt_metal::Array4D& input_tensor_start,
    float value,
    bool use_multicore = true,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    std::string_view implementation = "auto");

}  // namespace ttnn
