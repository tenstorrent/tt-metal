// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {

constexpr uint32_t cb_max = tt::CBIndex::c_4;
constexpr uint32_t cb_weight = tt::CBIndex::c_5;
constexpr uint32_t cb_denominator = tt::CBIndex::c_6;
constexpr uint32_t cb_numerator = tt::CBIndex::c_7;
constexpr uint32_t cb_product = tt::CBIndex::c_8;
constexpr uint32_t cb_reciprocal = tt::CBIndex::c_9;
constexpr uint32_t cb_max_temp = tt::CBIndex::c_10;

void copy_tile_to(uint32_t source_cb, uint32_t source_tile, uint32_t output_cb) {
    DataflowBuffer source(source_cb);
    DataflowBuffer output(output_cb);
    source.wait_front(source_tile + 1);
    output.reserve_back(1);
    reconfig_data_format_srca(source_cb);
    copy_init(source_cb);
    tile_regs_acquire();
    copy_tile(source_cb, source_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(output_cb);
    pack_tile(0, output_cb);
    tile_regs_release();
    output.push_back(1);
}

void binary_to(
    uint32_t lhs_cb, uint32_t lhs_tile, uint32_t rhs_cb, uint32_t rhs_tile, uint32_t output_cb, uint32_t op) {
    DataflowBuffer lhs(lhs_cb);
    DataflowBuffer rhs(rhs_cb);
    DataflowBuffer output(output_cb);
    lhs.wait_front(lhs_tile + 1);
    rhs.wait_front(rhs_tile + 1);
    output.reserve_back(1);
    reconfig_data_format(lhs_cb, rhs_cb);
    if (op == 0) {
        add_init(lhs_cb, rhs_cb);
    } else if (op == 1) {
        sub_init(lhs_cb, rhs_cb);
    } else {
        mul_init(lhs_cb, rhs_cb);
    }
    tile_regs_acquire();
    if (op == 0) {
        add_tiles(lhs_cb, rhs_cb, lhs_tile, rhs_tile, 0);
    } else if (op == 1) {
        sub_tiles(lhs_cb, rhs_cb, lhs_tile, rhs_tile, 0);
    } else {
        mul_tiles(lhs_cb, rhs_cb, lhs_tile, rhs_tile, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(output_cb);
    pack_tile(0, output_cb);
    tile_regs_release();
    output.push_back(1);
}

void replace_with_sum(uint32_t accumulator_cb, uint32_t value_cb) {
    DataflowBuffer accumulator(accumulator_cb);
    DataflowBuffer value(value_cb);
    accumulator.wait_front(1);
    value.wait_front(1);
    reconfig_data_format(accumulator_cb, value_cb);
    add_init(accumulator_cb, value_cb);
    tile_regs_acquire();
    add_tiles(accumulator_cb, value_cb, 0, 0, 0);
    tile_regs_commit();
    accumulator.pop_front(1);
    accumulator.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(accumulator_cb);
    pack_tile(0, accumulator_cb);
    tile_regs_release();
    accumulator.push_back(1);
}

void maximum_to(uint32_t lhs_cb, uint32_t rhs_cb, uint32_t rhs_tile, uint32_t output_cb) {
    DataflowBuffer lhs(lhs_cb);
    DataflowBuffer rhs(rhs_cb);
    DataflowBuffer output(output_cb);
    lhs.wait_front(1);
    rhs.wait_front(rhs_tile + 1);
    output.reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format_srca(lhs_cb);
    copy_init(lhs_cb);
    copy_tile(lhs_cb, 0, 0);
    reconfig_data_format_srca(rhs_cb);
    copy_init(rhs_cb);
    copy_tile(rhs_cb, rhs_tile, 1);
    binary_max_tile_init();
    binary_max_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(output_cb);
    pack_tile(0, output_cb);
    tile_regs_release();
    output.push_back(1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t candidate_kv_cb = get_compile_time_arg_val(0);
    constexpr uint32_t candidate_score_cb = get_compile_time_arg_val(1);
    constexpr uint32_t pooled_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_tiles = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(candidate_score_cb, candidate_kv_cb, pooled_cb);
    DataflowBuffer candidate_kv(candidate_kv_cb);
    DataflowBuffer candidate_score(candidate_score_cb);
    DataflowBuffer max_values(cb_max);
    DataflowBuffer weights(cb_weight);
    DataflowBuffer denominator(cb_denominator);
    DataflowBuffer numerator(cb_numerator);
    DataflowBuffer product(cb_product);
    DataflowBuffer reciprocal(cb_reciprocal);
    DataflowBuffer max_temp(cb_max_temp);
    DataflowBuffer pooled(pooled_cb);

    for (uint32_t tile = 0; tile < output_tiles; ++tile) {
        candidate_kv.wait_front(8);
        candidate_score.wait_front(8);

        copy_tile_to(candidate_score_cb, 0, cb_max);
        for (uint32_t candidate = 1; candidate < 8; ++candidate) {
            max_values.wait_front(1);
            maximum_to(cb_max, candidate_score_cb, candidate, cb_max_temp);
            max_values.pop_front(1);
            copy_tile_to(cb_max_temp, 0, cb_max);
            max_temp.pop_front(1);
        }

        for (uint32_t candidate = 0; candidate < 8; ++candidate) {
            binary_to(candidate_score_cb, candidate, cb_max, 0, cb_weight, 1);
            weights.wait_front(1);
            reconfig_data_format_srca(cb_weight);
            copy_init(cb_weight);
            exp_tile_init<true>();
            tile_regs_acquire();
            copy_tile(cb_weight, 0, 0);
            exp_tile<true>(0);
            tile_regs_commit();
            weights.pop_front(1);
            weights.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_weight);
            pack_tile(0, cb_weight);
            tile_regs_release();
            weights.push_back(1);

            binary_to(candidate_kv_cb, candidate, cb_weight, 0, cb_product, 2);
            if (candidate == 0) {
                copy_tile_to(cb_weight, 0, cb_denominator);
                copy_tile_to(cb_product, 0, cb_numerator);
            } else {
                replace_with_sum(cb_denominator, cb_weight);
                replace_with_sum(cb_numerator, cb_product);
            }
            weights.pop_front(1);
            product.pop_front(1);
        }

        reciprocal.reserve_back(1);
        denominator.wait_front(1);
        reconfig_data_format_srca(cb_denominator);
        copy_init(cb_denominator);
        recip_tile_init();
        tile_regs_acquire();
        copy_tile(cb_denominator, 0, 0);
        recip_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_reciprocal);
        pack_tile(0, cb_reciprocal);
        tile_regs_release();
        reciprocal.push_back(1);

        binary_to(cb_numerator, 0, cb_reciprocal, 0, pooled_cb, 2);

        max_values.pop_front(1);
        denominator.pop_front(1);
        numerator.pop_front(1);
        reciprocal.pop_front(1);
        candidate_kv.pop_front(8);
        candidate_score.pop_front(8);
    }
}
