// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "matmul_compute.hpp"
#include "rmsnorm_compute.hpp"

// The row statistics and the normalisation of the next M block are done one tile row at a time between the K blocks of
// the gate/up matmul (statistics) and of the down matmul (scales, normalisation), so they hide behind the matmuls.
struct NormWork {
    RmsnormCbs cbs;
    RmsnormShape shape;
    uint32_t own_blocks;
    uint32_t max_own_blocks;

    uint32_t units() const { return own_blocks * shape.rows; }

    void stats(uint32_t unit) const {
        if (unit < units()) {
            rmsnorm_stats_row(cbs, shape, unit / shape.rows, unit % shape.rows);
        } else {
            rmsnorm_stats_finalize(cbs, shape);
        }
    }

    void scales() const { rmsnorm_row_scales(cbs, shape, shape.group_size); }

    void apply(uint32_t unit) const {
        const uint32_t row = unit % shape.rows;
        const bool last_overall = unit + 1 == units();
        rmsnorm_apply_row(cbs, shape, row, row == shape.rows - 1, last_overall);
        if (last_overall) {
            CircularBuffer out(cbs.out);
            for (uint32_t b = own_blocks; b < max_own_blocks; b++) {
                out.reserve_back(shape.rows * shape.block_tiles);
                out.push_back(shape.rows * shape.block_tiles);
            }
        }
    }
};

void kernel_main() {
    constexpr MatmulShape gate_up = MATMUL_SHAPE("mm1");
    constexpr MatmulShape down = MATMUL_SHAPE("mm2");
    const NormWork norm = {
        RMSNORM_CBS,
        {gate_up.M_block,
         gate_up.K_block,
         CT_ARG("group_size"),
         CT_ARG("norm_dest_tiles"),
         CT_ARG("norm_inv_width_bits"),
         CT_ARG("norm_eps_bits")},
        RT_ARG("own_blocks"),
        CT_ARG("max_own_blocks")};

    compute_kernel_hw_startup<SrcOrder::Reverse>(gate_up.cb_in0, gate_up.cb_in1, gate_up.cb_partial);

    CircularBuffer(norm.cbs.scaler).wait_front(1);
    for (uint32_t unit = 0; unit <= norm.units(); unit++) {
        norm.stats(unit);
    }
    norm.scales();
    for (uint32_t unit = 0; unit < norm.units(); unit++) {
        norm.apply(unit);
    }

    for (uint32_t m_block = 0; m_block < CT_ARG("M_blocks"); m_block++) {
        const bool has_next = m_block + 1 < CT_ARG("M_blocks");
        matmul_phase(gate_up, [&](uint32_t step) {
            if (has_next && step <= norm.units()) {
                pack_reconfig_l1_acc(0);
                norm.stats(step);
                return true;
            }
            return false;
        });
        matmul_phase(down, [&](uint32_t step) {
            if (has_next && step <= norm.units()) {
                pack_reconfig_l1_acc(0);
                if (step == 0) {
                    norm.scales();
                } else {
                    norm.apply(step - 1);
                }
                return true;
            }
            return false;
        });
    }
}
