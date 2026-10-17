// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::operations::data_movement {

// Float32 -> BFloat16 datacopies round to nearest even like ttnn.typecast: the input is unpacked to a 32-bit DEST
// and ROUND_FP32_TO_BF16 makes the compute kernel narrow it with the typecast SFPU step.
inline bool datacopy_rounds_fp32_to_bf16(tt::ARCH arch, tt::DataFormat input_format, tt::DataFormat output_format) {
    return arch != tt::ARCH::QUASAR && input_format == tt::DataFormat::Float32 &&
           output_format == tt::DataFormat::Float16_b;
}

}  // namespace ttnn::operations::data_movement
