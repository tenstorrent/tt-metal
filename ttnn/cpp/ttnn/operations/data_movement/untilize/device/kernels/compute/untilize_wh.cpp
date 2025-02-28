// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/untilize.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t block_size_col = get_compile_time_arg_val(0);
    const uint32_t block_size_row = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);

    untilize_init(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t b = 0; b < block_size_col * third_dim; ++b) {
        cb_wait_front(tt::CBIndex::c_0, block_size_row);
        cb_reserve_back(tt::CBIndex::c_16, block_size_row);

        untilize_block(tt::CBIndex::c_0, block_size_row, tt::CBIndex::c_16);

        cb_push_back(tt::CBIndex::c_16, block_size_row);
        cb_pop_front(tt::CBIndex::c_0, block_size_row);
    }
}
}  // namespace NAMESPACE
