// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_device_operation_types.hpp"

namespace ttnn::prim::untilize_helper {
uint32_t get_largest_divisor(uint32_t dividend, uint32_t starting_divisor, uint32_t divisor_factor) {
    for (uint32_t curr_div = starting_divisor; curr_div > 0; curr_div--) {
        if (dividend % (curr_div * divisor_factor) == 0) {
            return curr_div;
        }
    }
    return 1;
}
}  // namespace ttnn::prim::untilize_helper
