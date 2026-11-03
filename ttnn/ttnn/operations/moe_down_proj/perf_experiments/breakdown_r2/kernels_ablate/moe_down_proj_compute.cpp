// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj compute. Per wave:
//   open_partials: one reserve of the nominal fp32 partials block (core_row_tiles_max x core_n_tiles).
//   accumulate_round (per K round): one matmul_block per expert sub-block of this core's rows,
//     in0 = the sub-block's x rows (popped per call), in1 = the expert's w chunk (slot e of the round, transposed
//     tiles, caller-managed cumulative wait), packed TileRowMajor with packer L1 accumulation into partials rows
//     [r0, r0 + rows) (out_base_offset). A sub-block whose rows don't divide the preferred subblock height is cut
//     into a main part and a tail call.
//   release_round: pop the x remainder and all NUM_EXPERTS w chunks.
//   untilize_wave: push the partials, untilize this core's tile rows into cb_output_rows, pop the remainder.

#include <cstdint>
#ifdef ABL_NO_COMPUTE
#define SKIP_COMPUTE
#endif

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "moe_down_proj_plan.h"

namespace {

struct ExpertSlotOffset {
    uint32_t base;
    ALWI uint32_t operator()(uint32_t /*block*/) const { return base; }
};

struct Subblock {
    uint32_t h;
    uint32_t w;
};

// Largest-area subblock (taller on ties) with h | rows, w | width, h * w <= limit.
ALWI constexpr Subblock best_subblock(uint32_t rows, uint32_t width, uint32_t limit) {
    Subblock best{1, 1};
    for (uint32_t w = 1; w <= width; ++w) {
        if (width % w) {
            continue;
        }
        for (uint32_t h = 1; h * w <= limit && h <= rows; ++h) {
            if (rows % h) {
                continue;
            }
            const uint32_t area = h * w;
            if (area > best.h * best.w || (area == best.h * best.w && h > best.h)) {
                best = {h, w};
            }
        }
    }
    return best;
}

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_round = get_compile_time_arg_val(1);
    constexpr uint32_t cb_partials = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output_rows = get_compile_time_arg_val(3);
    constexpr uint32_t cb_counts_compute = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_row_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_waves = get_compile_time_arg_val(6);
    constexpr uint32_t num_m_groups = get_compile_time_arg_val(7);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_rounds = get_compile_time_arg_val(9);
    constexpr uint32_t core_row_tiles_max = get_compile_time_arg_val(10);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(11);

    const uint32_t group = get_arg_val<uint32_t>(0);

    constexpr uint32_t x_round_tiles = core_row_tiles_max * k_block_tiles;
    constexpr uint32_t w_chunk_tiles = k_block_tiles * core_n_tiles;
    constexpr uint32_t partials_tiles = core_row_tiles_max * core_n_tiles;
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_round, cb_w_round, cb_partials);

    CircularBuffer x_buf(cb_x_round);
    CircularBuffer w_buf(cb_w_round);
    CircularBuffer partials_buf(cb_partials);

    // plan_wave inputs: identical on all three TRISCs (mailbox-synchronized read).
    cb_wait_front(cb_counts_compute, 1);
    uint32_t counts[MOE_NUM_EXPERTS];
    for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
        counts[e] = read_tile_value(cb_counts_compute, 0, e);
    }
    const MoePlan plan(counts, tensor_row_tiles, num_waves, num_m_groups);

    // One tile row, widest w: at 2x2 the partials pack serializes with the matmul, at 1 x w it hides behind it.
    // constexpr: a runtime shape costs ~45% more payload at 1 x w (per-call LLK scalar work).
    constexpr Subblock preferred = best_subblock(1, core_n_tiles, dest_limit);

    for (uint32_t wave = 0; wave < num_waves; ++wave) {
        if (plan.wave_empty(wave)) {
            continue;
        }
        const uint32_t row_lo = plan.group_first(wave, group);
        const uint32_t my_rows = plan.group_rows(wave, group);
        const uint32_t row_hi = row_lo + my_rows;

        // open_partials
        partials_buf.reserve_back(partials_tiles);

        // matmul_block_init carries the subblock shape; re-init only on a change. The first call of a wave
        // follows the untilize (or boot), so it also reconfigures both data formats.
        bool first_call = true;
        Subblock init_sb{0, 0};

        // Last expert slot this core reads; the writer pushes a whole round at once, so waiting for it up front
        // only separates the input wait from the matmul work in the zones.
        uint32_t last_slot = 0;
        for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
            uint32_t first;
            if (plan.expert_rows_in(e, row_lo, row_hi, &first) != 0) {
                last_slot = e + 1;
            }
        }

        {
            MaybeDeviceZoneScope("compute_accumulate");
            for (uint32_t round = 0; round < num_k_rounds; ++round) {
                const bool seeded = round > 0;
                if (my_rows > 0) {
                    MaybeDeviceZoneScope("compute_wait_inputs");
                    {
                        MaybeDeviceZoneScope("compute_wait_x");
                        x_buf.wait_front(my_rows * k_block_tiles);
                    }
                    w_buf.wait_front(last_slot * w_chunk_tiles);
                }
                for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
                    uint32_t first;
                    const uint32_t rows = plan.expert_rows_in(e, row_lo, row_hi, &first);
                    if (rows == 0) {
                        continue;
                    }
                    // in1 is NoWaitNoPop: cumulative wait up to this expert's slot of the round.
                    w_buf.wait_front((e + 1) * w_chunk_tiles);
                    const ExpertSlotOffset slot_offset{e * w_chunk_tiles};

                    const uint32_t main_rows = rows - rows % preferred.h;
                    uint32_t call_r0 = first - row_lo;
                    for (uint32_t part = 0; part < 2; ++part) {
                        const uint32_t call_rows = part == 0 ? main_rows : rows - main_rows;
                        if (call_rows == 0) {
                            continue;
                        }
                        const Subblock sb = part == 0 ? preferred : best_subblock(call_rows, core_n_tiles, dest_limit);
                        const auto shape =
                            MatmulBlockShape::of(call_rows / sb.h, core_n_tiles / sb.w, sb.h, sb.w, k_block_tiles, 1);
                        const uint32_t out_base = call_r0 * core_n_tiles;
                        const bool reinit = sb.h != init_sb.h || sb.w != init_sb.w;
                        init_sb = sb;

                        // Template order: transpose, packer_l1_acc, target, tile order, init, in0/in1 policy, hooks,
                        // untilize dim, inner dim fn, in0 source, in1 offset, caller_owns, seeded, activation,
                        // reconfig.
#define MOE_MATMUL(INIT, SEEDED, RECONFIG)            \
    matmul_block<                                     \
        true,                                         \
        true,                                         \
        LastBlockTarget::Interm,                      \
        OutputCBLayout::TileRowMajor,                 \
        matmul_config::InitMode::INIT,                \
        InputPolicy::WaitAndPopPerKBlock,             \
        InputPolicy::NoWaitNoPop,                     \
        NoPostCompute,                                \
        NoPreKBlock,                                  \
        NoPostKBlock,                                 \
        0,                                            \
        NoKBlockInnerDimFn,                           \
        NoIn0Source,                                  \
        ExpertSlotOffset,                             \
        true,                                         \
        SEEDED,                                       \
        NoneActivation,                               \
        matmul_config::DataFormatReconfig::RECONFIG>( \
        x_buf,                                        \
        w_buf,                                        \
        partials_buf,                                 \
        partials_buf,                                 \
        shape,                                        \
        {},                                           \
        {},                                           \
        core_n_tiles,                                 \
        core_n_tiles,                                 \
        {},                                           \
        {},                                           \
        {},                                           \
        slot_offset,                                  \
        out_base)

                        if (first_call) {
                            // Round 0 by construction (first sub-block of the wave).
                            MOE_MATMUL(Short, false, INPUT_AND_OUTPUT);
                        } else if (reinit) {
                            if (seeded) {
                                MOE_MATMUL(Short, true, NONE);
                            } else {
                                MOE_MATMUL(Short, false, NONE);
                            }
                        } else {
                            if (seeded) {
                                MOE_MATMUL(None, true, NONE);
                            } else {
                                MOE_MATMUL(None, false, NONE);
                            }
                        }
#undef MOE_MATMUL
                        first_call = false;
                        call_r0 += call_rows;
                    }
                }
                // release_round: the sub-block calls popped my_rows * k_block_tiles; the rest of the nominal round.
                x_buf.wait_front(x_round_tiles - my_rows * k_block_tiles);
                x_buf.pop_front(x_round_tiles - my_rows * k_block_tiles);
                w_buf.wait_front(MOE_NUM_EXPERTS * w_chunk_tiles);
                w_buf.pop_front(MOE_NUM_EXPERTS * w_chunk_tiles);
            }
        }
        // The partials pack left packer L1 accumulation on; the untilize must overwrite its output.
        pack_reconfig_l1_acc(0);

        // untilize_wave
        partials_buf.push_back(partials_tiles);
        if (my_rows > 0) {
            MaybeDeviceZoneScope("compute_untilize");
            untilize<core_n_tiles, cb_partials, cb_output_rows>(my_rows);
        }
        const uint32_t remainder = partials_tiles - my_rows * core_n_tiles;
        partials_buf.wait_front(remainder);
        partials_buf.pop_front(remainder);
    }
    cb_pop_front(cb_counts_compute, 1);
}
