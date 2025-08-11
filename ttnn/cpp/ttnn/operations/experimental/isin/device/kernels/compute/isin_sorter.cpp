// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "../isin_common.hpp"

#include "ckernel_sfpu.h"

#include <cstdint>

namespace NAMESPACE {

void MAIN {
    constexpr auto ctas = get_ctas();

    // read from elements and test_elements
    // pass to ckernel::topk_local_sort
    // pack to elements_sorted and test_elements_sorted
}

}  // namespace NAMESPACE
