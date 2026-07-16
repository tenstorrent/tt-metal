// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_supported.hpp"

namespace ttnn::operations::data_movement::untilize_codegen {

bool supported_by_codegen(const Tensor& /*input*/, const tt::tt_metal::MemoryConfig& /*output_mem_config*/) {
    return false;
}

}  // namespace ttnn::operations::data_movement::untilize_codegen
