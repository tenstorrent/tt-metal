// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace {

// DST slots. The numerator and denominator stay resident across all 8 candidates, so an output tile
// costs one pack rather than one per candidate.
constexpr uint32_t kNumerator = 0;
constexpr uint32_t kDenominator = 1;
constexpr uint32_t kWeight = 2;

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(dfb::candidate_score, dfb::candidate_kv, dfb::pooled);
    const uint32_t output_tiles = get_arg(args::output_tiles);
    DataflowBuffer candidate_kv(dfb::candidate_kv);
    DataflowBuffer candidate_score(dfb::candidate_score);
    // ca_bias / cb_bias: the position bias of the Ca and Cb halves of the projection.
    DataflowBuffer ca_bias(dfb::ca_bias);
    DataflowBuffer cb_bias(dfb::cb_bias);
    // The running max is the only intermediate that has to leave DST, because sub_reuse_dest_tiles takes
    // its second operand from a dataflow buffer.
    DataflowBuffer max_values(dfb::max_values);
    DataflowBuffer pooled(dfb::pooled);

    // Every row of candidate tile c carries slot c % 4, so that candidate's whole position bias is a
    // single row of the bias tensor and a row broadcast applies it. Ca candidates take their bias from
    // the left half of the projection, Cb candidates from the right.
    for (uint32_t tile = 0; tile < output_tiles; ++tile) {
        candidate_kv.wait_front(8);
        candidate_score.wait_front(8);
        ca_bias.wait_front(1);
        cb_bias.wait_front(1);

        // Pass 1: the running max over the 8 biased scores, reduced entirely inside DST.
        max_values.reserve_back(1);
        tile_regs_acquire();
        for (uint32_t candidate = 0; candidate < 8; ++candidate) {
            const uint32_t bias_dfb =
                candidate < 4 ? static_cast<uint32_t>(dfb::ca_bias) : static_cast<uint32_t>(dfb::cb_bias);
            const uint32_t destination = candidate == 0 ? 0 : 1;
            add_bcast_rows_init(dfb::candidate_score, bias_dfb);
            add_tiles_bcast_rows(dfb::candidate_score, bias_dfb, candidate, 0, destination, candidate & 3);
            if (candidate > 0) {
                binary_max_tile_init();
                binary_max_tile(0, 1, 0);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::max_values);
        tile_regs_release();
        max_values.push_back(1);

        // Pass 2: numerator and denominator of the softmax-weighted pool, also entirely inside DST.
        max_values.wait_front(1);
        pooled.reserve_back(1);
        tile_regs_acquire();
        for (uint32_t candidate = 0; candidate < 8; ++candidate) {
            const uint32_t bias_dfb =
                candidate < 4 ? static_cast<uint32_t>(dfb::ca_bias) : static_cast<uint32_t>(dfb::cb_bias);
            add_bcast_rows_init(dfb::candidate_score, bias_dfb);
            add_tiles_bcast_rows(dfb::candidate_score, bias_dfb, candidate, 0, kWeight, candidate & 3);
            sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::max_values);
            sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::max_values, 0, kWeight);
            exp_tile_init<true>();
            exp_tile<true>(kWeight);

            if (candidate == 0) {
                copy_dest_values_init();
                copy_dest_values<DataFormat::Float32>(kWeight, kDenominator);
            } else {
                add_binary_tile_init();
                add_binary_tile(kDenominator, kWeight, kDenominator);
            }

            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::candidate_kv);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::candidate_kv, candidate, kWeight);

            if (candidate == 0) {
                copy_dest_values_init();
                copy_dest_values<DataFormat::Float32>(kWeight, kNumerator);
            } else {
                add_binary_tile_init();
                add_binary_tile(kNumerator, kWeight, kNumerator);
            }
        }
        recip_tile_init();
        recip_tile(kDenominator);
        mul_binary_tile_init();
        mul_binary_tile(kNumerator, kDenominator, kNumerator);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(kNumerator, dfb::pooled);
        tile_regs_release();
        pooled.push_back(1);

        max_values.pop_front(1);
        candidate_kv.pop_front(8);
        candidate_score.pop_front(8);
        ca_bias.pop_front(1);
        cb_bias.pop_front(1);
    }
}
