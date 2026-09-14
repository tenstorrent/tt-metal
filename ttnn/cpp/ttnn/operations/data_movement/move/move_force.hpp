// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only native entrypoint. Keep independent of public dispatch so
// future codegen routing cannot silently turn a native comparison into codegen.
// Bound only in ttnn._ttnn.operations.data_movement, outside the public API.
Tensor move_force_native(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::data_movement::detail
