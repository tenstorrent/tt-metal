// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/kernel_structs.h"

#include <type_traits>

namespace ttnn::operations::experimental::unique {

using namespace tt;

enum class UniqueCB : std::underlying_type_t<CBIndex> {
    INPUT = CBIndex::c_0,
    INPUT_COMPARE = CBIndex::c_1,
    FIRST_OCCURRENCES_READ = CBIndex::c_2,
    FIRST_OCCURRENCES_WRITE = CBIndex::c_3,
    FIRST_OCCURRENCES_OUTPUT = CBIndex::c_4,
    RESULT_ACC = CBIndex::c_5,
    OUTPUT = CBIndex::c_6,
    OUTPUT_SIZE = CBIndex::c_7
};

}  // namespace ttnn::operations::experimental::unique
