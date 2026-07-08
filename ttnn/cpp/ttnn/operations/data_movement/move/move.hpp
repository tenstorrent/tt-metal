// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include "ttnn/types.hpp"

namespace ttnn {

// `implementation` selects the backing prim: "auto" (default) routes in-scope calls to
// `prim::move_codegen` and everything else to native via `supported_by_codegen()`; "native" always
// uses the existing native prim; "codegen" forces `prim::move_codegen`, which TT_FATALs if the call
// is out of scope. See codegen/move_codegen_supported.hpp.
Tensor move(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::string& implementation = "auto");

}  // namespace ttnn
