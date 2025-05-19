// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <type_traits>

#include "hostdevcommon/kernel_structs.h"

namespace ttnn::operations::experimental::scatter {

using namespace tt;

enum class ScatterReductionType : uint8_t { ADD, MULTIPLY, AMIN, AMAX };

enum class ScatterCB : std::underlying_type_t<tt::CBIndex> {
    INPUT = CBIndex::c_0,
    SRC = CBIndex::c_1,
    INDEX = CBIndex::c_2,
    DST = CBIndex::c_3
};

}  // namespace ttnn::operations::experimental::scatter
