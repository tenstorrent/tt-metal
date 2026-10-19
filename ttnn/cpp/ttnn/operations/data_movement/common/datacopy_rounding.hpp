// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::operations::data_movement {

// Float32 -> BFloat16 datacopies run with a 32-bit DEST so the packer rounds to nearest (ties away from zero);
// with a 16-bit DEST the value was truncated on the way in.
inline bool datacopy_rounds_fp32_to_bf16(tt::ARCH arch, tt::DataFormat input_format, tt::DataFormat output_format) {
    return arch != tt::ARCH::QUASAR && input_format == tt::DataFormat::Float32 &&
           output_format == tt::DataFormat::Float16_b;
}

}  // namespace ttnn::operations::data_movement
