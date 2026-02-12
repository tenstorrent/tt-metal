// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

inline constexpr uint32_t READ_ALIGNMENT = 64;

struct RepeatSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::CoreRange total_cores{tt::tt_metal::CoreCoord{0, 0}};
};

}  // namespace ttnn::prim
