// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_supported.hpp"

namespace ttnn::operations::data_movement::tilize_codegen {

bool supported_by_codegen(const Tensor&, const tt::tt_metal::MemoryConfig&) { return false; }

bool is_demoted(const Tensor&, const tt::tt_metal::MemoryConfig&) { return false; }

}  // namespace ttnn::operations::data_movement::tilize_codegen
