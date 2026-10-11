// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// The reader supplies, per channel tile, the four tap-shifted input tiles and, per channel block, diagonal tap
// weight tiles. Scaling a tile's columns by a tap is a matmul with that tap's diagonal, so the four taps
// accumulate in the destination registers with no partial round trip; SiLU then runs in place.
template <uint32_t block_ct, uint32_t Mt, uint32_t group>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    constexpr uint32_t tap_count = 4;
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer output(dfb::output);
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::act_tile, dfb::weights, dfb::output);
    // The SFPU init only claims its own address modifier, so both inits hold for the whole kernel.
    matmul_init(dfb::act_tile, dfb::weights);
    silu_tile_init();

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t block = work / Mt;
        if (item == 0 || (work - 1) / Mt != block) {
            weights.wait_front(tap_count * block_ct);
        }
        for (uint32_t ct0 = 0; ct0 < block_ct; ct0 += group) {
            activation.wait_front(group * tap_count);
            output.reserve_back(group);
            tile_regs_acquire();
            for (uint32_t i = 0; i < group; ++i) {
                for (uint32_t tap = 0; tap < tap_count; ++tap) {
                    matmul_tiles(dfb::act_tile, dfb::weights, i * tap_count + tap, tap * block_ct + ct0 + i, i);
                }
            }
            for (uint32_t i = 0; i < group; ++i) {
                silu_tile(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < group; ++i) {
                pack_tile(i, dfb::output);
            }
            tile_regs_release();
            output.push_back(group);
            activation.pop_front(group * tap_count);
        }
        if (item + 1 == wi_count || (work + 1) / Mt != block) {
            weights.pop_front(tap_count * block_ct);
        }
    }
}
