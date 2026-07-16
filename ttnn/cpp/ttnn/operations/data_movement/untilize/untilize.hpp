// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include "ttnn/types.hpp"

namespace ttnn {

// `implementation` selects the host dispatch: "auto" (default) routes in-scope, non-demoted
// cases to the codegen implementation and everything else to native; "native" always uses
// the existing native prim; "codegen" always uses the codegen prim (TT_FATALs if the case is
// unsupported). See codegen/untilize_codegen_supported.hpp.
ttnn::Tensor untilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool use_multicore = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::string& implementation = "auto");

}  // namespace ttnn
