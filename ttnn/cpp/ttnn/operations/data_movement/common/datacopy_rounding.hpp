// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::operations::data_movement {

inline bool datacopy_rounds_fp32_to_bf16(tt::ARCH arch, tt::DataFormat input_format, tt::DataFormat output_format) {
    return arch != tt::ARCH::QUASAR && input_format == tt::DataFormat::Float32 &&
           output_format == tt::DataFormat::Float16_b;
}

}  // namespace ttnn::operations::data_movement
