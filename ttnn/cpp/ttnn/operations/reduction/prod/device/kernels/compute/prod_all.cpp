// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr auto input_cb = tt::CBIndex::c_0;
    constexpr auto partial_prod_cb = tt::CBIndex::c_2;
    constexpr auto final_output_cb = tt::CBIndex::c_3;

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    binary_op_init_common(input_cb, partial_prod_cb, final_output_cb);

    // Reduction by running product across num_tiles. The original kernel kept
    // DST alive across iterations via pack-to-partial + reload pattern; the
    // chain reproduces the same dataflow via three stages:
    //   Stage 1 (seed): input[0] -> partial_prod
    //   Stage 2 (accumulator, n_tiles=num_tiles-1):
    //       partial_prod *= input[t]  (BinaryFpu + PackTile<partial_prod>)
    //   Stage 3 (final): partial_prod -> final_output
    //
    // Reconfig audit: original used reconfig_data_format(input, partial_prod)
    // + pack_reconfig_data_format(final_output) once at boot, then plain
    // copy_tile / mul_tiles / pack_tile (no per-call _with_dt). Chain emits
    // Input + Output reconfigs on the boot-to-first-call transition (folded
    // by prev-CB elision when formats match the boot state) so the effective
    // sequence matches.
    if constexpr (num_tiles == 1) {
        // Single tile: copy direct to final.
        compute_kernel_lib::copy<input_cb, final_output_cb>(1u);
    } else {
        compute_kernel_lib::copy<input_cb, partial_prod_cb>(1u);
        compute_kernel_lib::mul<input_cb, partial_prod_cb, partial_prod_cb>(num_tiles - 1u);
        compute_kernel_lib::copy<partial_prod_cb, final_output_cb>(1u);
    }
}
