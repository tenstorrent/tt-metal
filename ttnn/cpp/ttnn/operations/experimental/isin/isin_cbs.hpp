// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/kernel_structs.h"

#include <type_traits>

namespace ttnn::operations::experimental::isin {

enum class IsInCB : std::underlying_type_t<tt::CBIndex> {
    ELEMENTS = tt::CBIndex::c_0,
    TEST_ELEMENTS = tt::CBIndex::c_1,
    OUTPUT = tt::CBIndex::c_2
};

}  // namespace ttnn::operations::experimental::isin
