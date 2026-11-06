// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilizes one page at a time: the D/32 tiles of a (b, h, s_blk) become 32
// rows of D values, which is the page the writer stores as is.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

constexpr uint32_t in_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t out_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t TILES_PER_PAGE = get_compile_time_arg_val(2);

void kernel_main() {
    const uint32_t num_pages = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(in_cb_index, out_cb_index);
    compute_kernel_lib::untilize<
        TILES_PER_PAGE,
        in_cb_index,
        out_cb_index,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(num_pages);
}
