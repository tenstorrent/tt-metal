// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_route_dispatch compute. One block per phase (the whole x shard: rps tile rows x ht tiles):
//   token core: partial_logits_block (raw matmul_block, K = ht over the faces-only W slot; first, as soon as W lands)
//               -> packed into this member's region of cb_partials
//               -> logits_block + softmax_block fused on the SFPU (p_0 [+ p_1 [+ p_2]], max, exp, 1 / sum, mul over the
//               E expert columns, exact fp32) -> cb_scores
//               -> untilize_x_block (in the select shadow)
//   every core: tilize_gathered_block (all owned output blocks of 32 rows x w tiles, into the output shard)

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "nomic_moe_route_dispatch_softmax8.hpp"
#include "nomic_moe_route_dispatch_w_layout.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t ht = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t rps = get_named_compile_time_arg_val("rows_per_shard");
    constexpr uint32_t G = get_named_compile_time_arg_val("hidden_split");
    constexpr uint32_t w = get_named_compile_time_arg_val("block_width_tiles");
    constexpr uint32_t weight_tiles = get_named_compile_time_arg_val("weight_tiles");
    constexpr uint32_t cb_x_tiles = get_named_compile_time_arg_val("cb_x_tiles");
    constexpr uint32_t cb_router_weight = get_named_compile_time_arg_val("cb_router_weight");
    constexpr uint32_t cb_own_partial = get_named_compile_time_arg_val("cb_own_partial");
    constexpr uint32_t cb_own_partial_west = get_named_compile_time_arg_val("cb_own_partial_west");
    constexpr uint32_t cb_w_faces = get_named_compile_time_arg_val("cb_w_faces");
    constexpr uint32_t w_slice_nocs = get_named_compile_time_arg_val("w_slice_nocs");
    constexpr uint32_t cb_partials = get_named_compile_time_arg_val("cb_partials");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_code_stage = get_named_compile_time_arg_val("cb_code_stage");
    constexpr uint32_t cb_x_rows = get_named_compile_time_arg_val("cb_x_rows");
    constexpr bool push_gather = get_named_compile_time_arg_val("push_gather") != 0;
    constexpr uint32_t stage_tiles = get_named_compile_time_arg_val("stage_tiles");
    constexpr uint32_t cb_stage_tiles = get_named_compile_time_arg_val("cb_stage_tiles");
    constexpr uint32_t cb_gathered_rows = get_named_compile_time_arg_val("cb_gathered_rows");
    constexpr uint32_t cb_dispatched = get_named_compile_time_arg_val("cb_dispatched");
    static_assert(G >= 1 && G <= 3, "whole rows (interleaved x) or a 2- or 3-slice nd_norm hidden split");

    const bool is_token = get_arg_val<uint32_t>(0) != 0;
    const uint32_t member = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const bool stages = get_arg_val<uint32_t>(3) != 0;  // this member pushes rows (stages a tile range)

    DataflowBuffer x_tiles(cb_x_tiles);
    DataflowBuffer router_weight(cb_router_weight);
    DataflowBuffer own_partial(cb_own_partial);
    DataflowBuffer own_partial_west(cb_own_partial_west);
    DataflowBuffer partials(cb_partials);
    DataflowBuffer scores(cb_scores);
    DataflowBuffer x_rows(cb_x_rows);
    DataflowBuffer gathered(cb_gathered_rows);
    DataflowBuffer dispatched(cb_dispatched);

    if (is_token) {
        const auto split = G > 1 ? route_dispatch::half_split(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5))
                                 : route_dispatch::HalfSplit{0, 0};
        const route_dispatch::TileOrder<G, rps> tile_order(split);
        // Matmul takes in0 (x) on SrcB and in1 (W) on SrcA. in1 is the faces-only W through the 1 KB-page view
        // cb_w_faces; the partial is packed straight into this member's region of cb_partials (the reader / writer
        // send it to the peers from there).
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_tiles, cb_w_faces, cb_partials);

        // ---- partial_logits_block: x[rps x ht] @ W[slice member] -> rps fp32 tiles ----
        {
            MaybeDeviceZoneScope("cmp_wait_w");
            router_weight.wait_front(weight_tiles);
        }
        {
            MaybeDeviceZoneScope("cmp_matmul");
            // Raw matmul_block, bypassing ckl::matmul_block: the compact W layout needs a non-linear in1 tile index
            // (w_face_tile) and the output goes to a region of a CB this thread does not produce. One partial tile at a
            // time (rt_dim = 1, K inner) in tile_order, each published to both DM RISCs as soon as it is packed (the
            // writer sends it to the peers, the reader publishes it to the softmax once the peers' tiles landed).
            // Same per-tile MVMULs and K accumulation order as one rt_dim = rps block: bit-identical expert columns.
            const uint32_t slot = route_dispatch::w_slot(member, G, w_slice_nocs);
            ckernel::matmul_block_init(cb_x_tiles, cb_w_faces, false, 1, 1, ht);
            x_tiles.wait_front(rps * ht);
            for (uint32_t rank = 0; rank < rps; ++rank) {
                const uint32_t tile = tile_order.order[rank];
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < ht; ++kt) {
                    ckernel::matmul_block(
                        cb_x_tiles,
                        cb_w_faces,
                        tile * ht + kt,
                        route_dispatch::w_face_tile(slot, ht, kt),
                        0,
                        false,
                        1,
                        1,
                        ht);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_partials, member * rps + tile);
                tile_regs_release();
                own_partial.reserve_back(1);
                own_partial.push_back(1);
                own_partial_west.reserve_back(1);
                own_partial_west.push_back(1);
            }
            x_tiles.pop_front(rps * ht);
        }
        router_weight.pop_front(weight_tiles);

        // ---- logits_block + softmax_block + top-2: one DEST window a tile, exact fp32 on the SFPU
        // (nomic_moe_route_dispatch_softmax8.hpp: contributor sum (p_0 + p_1) + p_2, row max, exp, 1 / sum, mul over
        // the 8 expert columns, top-2 code + weights) -> cb_scores ----
        {
            MaybeDeviceZoneScope("cmp_wait_partials");
            partials.wait_front(G);  // the first tile in tile_order; the others are waited per pass
        }
        {
            // Split softmax, G > 1: one pass a half of this member's share, in the order the select
            // RISCs need them (w_layout.hpp half_at_rank), each packed into its own cb_code_stage page and published
            // at once (the DM RISCs send its face to the group). G = 1: every tile, published one at a time.
            MaybeDeviceZoneScope("cmp_softmax");
            reconfig_data_format_srca(cb_partials);
            pack_reconfig_data_format(cb_scores);
            copy_init(cb_partials);
            MATH((route_dispatch_softmax8::init()));
            if constexpr (G == 1) {
                for (uint32_t tile = 0; tile < rps; ++tile) {
                    partials.wait_front(G * (tile + 1));
                    scores.reserve_back(1);
                    tile_regs_acquire();
                    copy_tile(cb_partials, tile, 0);
                    MATH((_llk_math_eltwise_unary_sfpu_params_(route_dispatch_softmax8::face<G>, 0, VectorMode::C)));
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_scores);
                    tile_regs_release();
                    scores.push_back(1);
                }
            } else {
                DataflowBuffer code_stage(cb_code_stage);
                for (uint32_t pos = member; pos < split.hn; pos += G) {
                    const uint32_t half = route_dispatch::half_at_rank(pos, split);
                    const uint32_t tile = half / 2;
                    partials.wait_front(G * (tile_order.slot[tile] + 1));
                    code_stage.reserve_back(1);
                    tile_regs_acquire();
                    for (uint32_t contributor = 0; contributor < G; ++contributor) {
                        copy_tile(cb_partials, contributor * rps + tile, contributor);
                    }
                    MATH((route_dispatch_softmax8::share_faces<G>(half % 2 == 0, half % 2 == 1)));
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_code_stage);
                    tile_regs_release();
                    code_stage.push_back(1);
                }
            }
        }

        // ---- untilize_x_block: x round 2, the whole shard in one call (whole-piece push: the staged range), in the
        // select shadow: the RISCs select while it runs, and the push needs it only after the count table ----
        if constexpr (push_gather) {
            if (stages) {
                DataflowBuffer stage(cb_stage_tiles);
                {
                    MaybeDeviceZoneScope("cmp_wait_stage");
                    stage.wait_front(rps * stage_tiles);
                }
                MaybeDeviceZoneScope("cmp_untilize");
                ckl::untilize<stage_tiles, cb_stage_tiles, cb_x_rows>(stage, x_rows, rps);
            }
        } else {
            {
                MaybeDeviceZoneScope("cmp_wait_stage");
                x_tiles.wait_front(rps * ht);
            }
            MaybeDeviceZoneScope("cmp_untilize");
            ckl::untilize<ht, cb_x_tiles, cb_x_rows>(x_tiles, x_rows, rps);
        }
        // The passes may not have reached the trailing tiles of tile_order; waiting here keeps that off the untilize.
        partials.wait_front(G * rps);
        partials.pop_front(G * rps);

    } else {
        compute_kernel_hw_startup(cb_gathered_rows, cb_dispatched);
    }

    // ---- tilize_gathered_block: every owned block in one call, into the output shard ----
    if (num_blocks > 0) {
        {
            MaybeDeviceZoneScope("cmp_wait_land");
            gathered.wait_front(w);  // the owned blocks: their push writes have landed here (reader owner_land)
        }
        MaybeDeviceZoneScope("cmp_tilize");
        ckl::tilize<w, cb_gathered_rows, cb_dispatched>(gathered, dispatched, num_blocks);
    }
}
