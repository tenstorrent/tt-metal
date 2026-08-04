// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string_view>
#include "ttnn/types.hpp"

namespace ttnn {

// `implementation` selects which prim this dispatches to: "auto" (default) routes to
// prim::gather_codegen when ttnn::operations::data_movement::gather::supported_by_codegen() accepts
// the call and it is not perf-demoted; "native" always uses prim::gather; "codegen" always uses
// prim::gather_codegen (TT_FATAL if unsupported). See gather/codegen/gather_codegen_supported.hpp.
Tensor gather(
    const Tensor& input_tensor,
    int8_t dim,
    const Tensor& input_index_tensor,
    bool sparse_grad,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    std::string_view implementation = "auto");

}  // namespace ttnn
