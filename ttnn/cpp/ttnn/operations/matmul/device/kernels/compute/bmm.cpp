// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul_op.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    ckernel::MatmulOpConfig cfg{};
    cfg.in0_cb_id = cb_in0;
    cfg.in1_cb_id = cb_in1;
    cfg.out_cb_id = cb_out;

    ckernel::TileMatmulOp mm(cfg);
    mm.init();
    mm.run(
        batch,
        Mt,
        Nt,
        Kt,
        /*in0_num_subblocks=*/1,
        /*in1_num_subblocks=*/1,
        /*in0_block_num_tiles=*/1,
        /*in1_block_num_tiles=*/1,
        /*in1_block_w=*/1);
}
