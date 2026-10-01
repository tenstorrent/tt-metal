// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_sum_obj(dfb::sum);

    compute_kernel_hw_startup(dfb::y, dfb::scaler, dfb::dx);

    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                if (h == 0) {
                    mask_tile_to_dfb<dfb::dy, dfb::mask, dfb::add>(
                        /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);
                } else {
                    // The y*dy buffer under a second name; one FIFO, not an extra buffer.
                    constexpr auto dfb_inter0_id = dfb::ydy;
                    mask_tile_to_dfb<dfb::dy, dfb::mask, dfb_inter0_id>(
                        /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);

                    add_tiles_to_dfb<dfb::add, dfb_inter0_id, dfb::add>();
                }
            } else {
                if (h == 0) {
                    copy_tile_to_dfb<dfb::dy, dfb::add>();
                } else {
                    add_tiles_to_dfb<dfb::add, dfb::dy, dfb::add>();
                }
            }
        }

        ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, dfb::add, dfb::scaler, dfb::sum>(
            ckl::ReduceInputBlockShape::single());

        for (uint32_t h = 0; h < Ht; ++h) {
            constexpr auto dfb_exp_id = dfb::ydy;  // the y * dy buffer, reused to hold exp(y)
            exp_tile_to_dfb<dfb::y, dfb_exp_id>();

            // sum * exp(y)
            mul_tiles_bcast_rows_to_dfb<dfb_exp_id, dfb::sum, dfb::dy_m_sum>(0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_dfb<dfb::dy, dfb::dy_m_sum, dfb::dx>();
        }

        dfb_sum_obj.pop_front(onetile);
#else

        // step 1, compute y * dy
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                mul_tiles_and_mask_tile_to_dfb<dfb::y, dfb::dy, dfb::mask, dfb::ydy>(
                    0, 0, 0, /*pop0=*/1, /*pop1=*/1, /*popm=*/0);
            } else {
                mul_tiles_to_dfb<dfb::y, dfb::dy, dfb::ydy>();
            }

            if (h == 0) {
                copy_tile_to_dfb<dfb::ydy, dfb::add>();
            } else {
                add_tiles_to_dfb<dfb::add, dfb::ydy, dfb::add>();
            }
        }

        // step 2, compute sum(y * dy)
        ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, dfb::add, dfb::scaler, dfb::sum>(
            ckl::ReduceInputBlockShape::single());

        // step 3, compute final result
        for (uint32_t h = 0; h < Ht; ++h) {
            // dy - sum
            sub_tiles_bcast_rows_to_dfb<dfb::dy, dfb::sum, dfb::dy_m_sum>(0, 0, /*pop0=*/1, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_dfb<dfb::y, dfb::dy_m_sum, dfb::dx>();
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_dfb<dfb::y, dfb::dy_m_sum, dfb::dx>();
#endif
        }

        dfb_sum_obj.pop_front(onetile);
#endif
    }
}
