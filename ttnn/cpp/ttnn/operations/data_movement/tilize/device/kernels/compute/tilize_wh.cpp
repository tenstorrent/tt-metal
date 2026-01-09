// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
// #include "api/debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t block_size_col = get_compile_time_arg_val(0);
    const uint32_t block_size_row = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    tilize_init(tt::CBIndex::c_0, block_size_row, tt::CBIndex::c_16);

    DPRINT_MATH(DPRINT << "this is the math kernel" << ENDL());
    DPRINT_PACK(DPRINT << "this is the pack kernel" << ENDL());
    DPRINT_UNPACK(DPRINT << "this is the unpack kernel" << ENDL());
    DPRINT_DATA0(DPRINT << "this is the data movement kernel on noc 0" << ENDL());
    DPRINT_DATA1(DPRINT << "this is the data movement kernel on noc 1" << ENDL());

    for (uint32_t b = 0; b < block_size_col * third_dim; ++b) {
        cb_wait_front(tt::CBIndex::c_0, block_size_row);
        cb_reserve_back(tt::CBIndex::c_16, block_size_row);

        DPRINT_UNPACK({ DPRINT << "cb_in is: " << TSLICE(tt::CBIndex::c_0, 0, SliceRange::h0_w0_32()) << ENDL(); });

        tilize_block(tt::CBIndex::c_0, block_size_row, tt::CBIndex::c_16);

        DPRINT_UNPACK({ DPRINT << "cb_out is: " << TSLICE(tt::CBIndex::c_16, 0, SliceRange::h0_w0_32()) << ENDL(); });
        cb_push_back(tt::CBIndex::c_16, block_size_row);
        cb_pop_front(tt::CBIndex::c_0, block_size_row);
    }
    tilize_uninit(tt::CBIndex::c_0, tt::CBIndex::c_16);
}
}  // namespace NAMESPACE
