// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/kernel_structs.h"

#include <type_traits>

namespace ttnn::operations::experimental::isunique {

using namespace tt;

enum class IsUniqueCB : std::underlying_type_t<CBIndex> {
    INPUT = CBIndex::c_0,
    INDEX_HINT = CBIndex::c_1,
    FIRST_OCCURRENCES = CBIndex::c_2,
    OUTPUT = CBIndex::c_3
};

}  // namespace ttnn::operations::experimental::isunique
