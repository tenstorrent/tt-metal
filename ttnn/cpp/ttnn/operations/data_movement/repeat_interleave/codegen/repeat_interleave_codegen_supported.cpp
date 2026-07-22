// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"

namespace ttnn::operations::data_movement {

bool supported_by_codegen(
    const Tensor& /*input*/,
    uint32_t /*repeats*/,
    int32_t /*dim*/,
    const std::optional<MemoryConfig>& /*output_mem_config*/) {
    return false;
}

}  // namespace ttnn::operations::data_movement
