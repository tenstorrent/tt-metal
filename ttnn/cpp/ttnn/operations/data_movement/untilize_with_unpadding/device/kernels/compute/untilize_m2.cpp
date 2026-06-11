// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-private copy of the shared untilize compute kernel. The legacy compute kernel
// (data_movement/untilize/device/kernels/compute/untilize.cpp) is still consumed positionally by the
// un-migrated untilize_with_unpadding variants (and by untilize itself), so the migrated single-core and
// default multi-core interleaved factories carry their own copy here. Only the binding mechanism changed:
// the src / out CB ids come from the DFB binding tokens (dfb::), and the two block-count scalars from the
// named compile-time arg namespace (args::). The untilize LLK call is preserved verbatim.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    constexpr uint32_t src_cb_id = dfb::src_cb_id;
    constexpr uint32_t out_cb_id = dfb::out_cb_id;

    compute_kernel_hw_startup(src_cb_id, out_cb_id);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        src_cb_id,
        out_cb_id,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}
