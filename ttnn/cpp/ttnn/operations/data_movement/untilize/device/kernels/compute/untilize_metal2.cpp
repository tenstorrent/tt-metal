// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of untilize.cpp (beside it). Carries the Metal 2.0 rewrite (named args + DFB
// bindings) so ops ported to Metal 2.0 can bind the untilize compute kernel without converting the
// legacy original in place (which would break its many still-legacy binders). Created by the
// data_movement/fold port (the first Metal 2.0 consumer); other Metal 2.0 consumers reuse it.
// Its binding names (dfb::src / dfb::out) and named args are the shared interface — do not rename.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

template <uint32_t per_core_block_cnt, uint32_t per_core_block_tile_cnt>
TT_KERNEL void untilize() {
    compute_kernel_hw_startup(dfb::src, dfb::out);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        dfb::src,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}
