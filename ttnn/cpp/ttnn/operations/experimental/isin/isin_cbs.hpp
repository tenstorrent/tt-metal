// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/kernel_structs.h"

#include <type_traits>

namespace ttnn::operations::experimental::isin {

using namespace tt;

enum class IsInCB : std::underlying_type_t<CBIndex> {
    ELEMENTS = CBIndex::c_0,
    TEST_ELEMENTS = CBIndex::c_1,
    OUTPUT = CBIndex::c_2
};

}  // namespace ttnn::operations::experimental::isin
