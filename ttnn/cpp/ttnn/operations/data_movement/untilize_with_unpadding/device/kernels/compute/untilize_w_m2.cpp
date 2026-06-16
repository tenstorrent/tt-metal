// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of data_movement/untilize/device/kernels/compute/untilize_w.cpp.
// The legacy source is shared (also used by untilize), so it is forked here (not edited in place) and
// ported to Metal 2.0 named bindings for untilize_with_unpadding's multi-core COL interleaved factory.
// Only the access mechanism changed:
//   - the input/output CB ids (legacy c_0 / c_16) come from the DFB tokens (dfb::in / dfb::out)
//   - per_core_block_cnt / per_core_block_tile_cnt / third_dim become named compile-time args (args::)
// The block-count arithmetic passed to the untilize helper is preserved verbatim.

#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    constexpr uint32_t third_dim = get_arg(args::third_dim);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    compute_kernel_lib::untilize<
        1,
        dfb::in,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
        per_core_block_cnt * per_core_block_tile_cnt * third_dim);
}
