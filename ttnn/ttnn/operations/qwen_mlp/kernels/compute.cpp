// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "matmul_compute.hpp"
#include "rmsnorm_compute.hpp"

void kernel_main() {
    constexpr RmsnormCbs norm_cbs = RMSNORM_CBS;
    constexpr MatmulShape gate_up = MATMUL_SHAPE("mm1");
    constexpr MatmulShape down = MATMUL_SHAPE("mm2");
    constexpr RmsnormShape norm_shape = {
        gate_up.M_block,
        gate_up.K_block,
        CT_ARG("group_size"),
        CT_ARG("norm_dest_tiles"),
        CT_ARG("norm_inv_width_bits"),
        CT_ARG("norm_eps_bits")};
    const uint32_t own_blocks = RT_ARG("own_blocks");
    CircularBuffer norm_out(norm_cbs.out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(gate_up.cb_in0, gate_up.cb_in1, gate_up.cb_partial);

    CircularBuffer(norm_cbs.scaler).wait_front(1);
    for (uint32_t m_block = 0; m_block < CT_ARG("M_blocks"); m_block++) {
        rmsnorm_partial_sums(norm_cbs, norm_shape, own_blocks);
        rmsnorm_row_scales(norm_cbs, norm_shape, norm_shape.group_size);
        rmsnorm_apply(norm_cbs, norm_shape, own_blocks);
        for (uint32_t b = own_blocks; b < CT_ARG("max_own_blocks"); b++) {
            norm_out.reserve_back(norm_shape.rows * norm_shape.block_tiles);
            norm_out.push_back(norm_shape.rows * norm_shape.block_tiles);
        }
        matmul_phase(gate_up);
        matmul_phase(down);
    }
}
