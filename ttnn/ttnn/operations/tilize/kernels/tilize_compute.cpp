// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize compute (TRISC0/1/2). Single compute phase: RM -> TILE.
//
// One compute_kernel_lib::tilize call covers all num_blocks blocks of
// 1 x chunk_wt tiles. The helper owns wait_front / reserve_back / LLK /
// push_back / pop_front per block and issues tilize_init / tilize_uninit once
// around the whole loop (InitAndUninit).
//
// RECONFIG: NoReconfigure on the no-cast path. compute_kernel_hw_startup(
// cb_rm_input, cb_tiled_output) has already programmed srcA/srcB from the input
// CB and the packer from the output CB, so a reconfigure would be pure
// redundant CFG traffic (dominant on ~1 us sharded cases).
//
// FP32MODE: decided inside the kernel from the CB format, no CT arg needed.
// Lossless is mandatory for fp32 input — fast tilize truncates fp32 -> tf32,
// which fails the bit-identity oracle. Lossless carries two static_asserts the
// host satisfies via fp32_dest_acc_en=True and
// unpack_to_dest_mode[cb_rm_input] = UnpackToDestFp32.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_input = 0;
    constexpr uint32_t cb_tiled_output = 16;

    constexpr uint32_t chunk_wt = get_compile_time_arg_val(0);
    constexpr uint32_t needs_cast = get_compile_time_arg_val(1);

    using namespace compute_kernel_lib::tilize_config;

    constexpr ReconfigureRegisterDatatypeMode reconfig_mode =
        needs_cast ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                   : ReconfigureRegisterDatatypeMode::NoReconfigure;

    constexpr Fp32Mode fp32_mode =
        compute_kernel_lib::is_fp32_input_format<cb_rm_input>() ? Fp32Mode::Lossless : Fp32Mode::Fast;

    compute_kernel_hw_startup(cb_rm_input, cb_tiled_output);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_lib::tilize<
        chunk_wt,
        cb_rm_input,
        cb_tiled_output,
        InitUninitMode::InitAndUninit,
        WaitMode::WaitBlock,
        reconfig_mode,
        fp32_mode>(num_blocks);
}
