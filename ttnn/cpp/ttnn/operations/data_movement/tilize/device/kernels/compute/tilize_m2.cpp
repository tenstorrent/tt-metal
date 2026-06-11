// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of tilize.cpp (op-private copy). The legacy compute kernel is still consumed positionally
// by tilize's un-migrated factories (block, height-sharded, width-sharded) and must not be touched, so the
// migrated interleaved factories (single-core, multi-core default) carry their own copy here. Only the
// binding mechanism changed: the CB ids come from the DFB binding tokens (dfb::), and the per-core block
// counts from named compile-time args (args::). The tilize LLK call is preserved verbatim.

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = dfb::cb_id_in0;
    constexpr uint32_t cb_id_out0 = dfb::cb_id_out0;
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    compute_kernel_hw_startup(cb_id_in0, cb_id_out0);

    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<cb_id_in0>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        per_core_block_tile_cnt,
        cb_id_in0,
        cb_id_out0,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(per_core_block_cnt);
}
