// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Abs, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto num_reduced_tiles_along_dim = get_arg_val<uint32_t>(i++);

    constexpr std::uint32_t cb_x = tt::CB::c_in0 + 0;    // input
    constexpr std::uint32_t cb_one = tt::CB::c_in0 + 1;  // one

    constexpr std::uint32_t cb_y = tt::CB::c_out0 + 0;   // output

    constexpr std::uint32_t cb_tmp0 = tt::CB::c_intermed0 + 0;
    constexpr std::uint32_t cb_tmp1 = tt::CB::c_intermed0 + 1;

    constexpr std::uint32_t cb_val = cb_tmp0;  // f(x)
    constexpr std::uint32_t cb_cal = cb_tmp1;  // calculate f(x) over dimensions

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    cb_wait_front(cb_one, onetile);  // comes from the reader

    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // PARTIAL migration: f(x) chain (Abs path only).
            //   migrated: !IS_ZERO branches via CopyTile + Abs + (Negative if MINUS_INF) + PackTile.
            //   skipped : IS_ZERO branches use unary_ne_tile (runtime-param UnaryNe — chain dispatch
            //             path doesn't support member-exec runtime-param ops in v1).
#ifdef IS_ZERO
            tile_regs_acquire();
            cb_wait_front(cb_x, onetile);  // comes from the reader
            cb_reserve_back(cb_val, onetile);

            copy_tile_init_with_dt(cb_x);
            copy_tile(cb_x, 0, dst0);

            unary_ne_tile_init();
            unary_ne_tile(dst0, 0);

#ifdef MINUS_INF
            negative_tile_init();
            negative_tile(dst0);
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_val);
            tile_regs_release();

            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_val, onetile);
#else
#if defined FP32_DEST_ACC_EN
            reconfig_data_format_srca(cb_x);
            pack_reconfig_data_format(cb_val);
#endif
            {
                using namespace compute_kernel_lib;
#ifdef MINUS_INF
                eltwise_chain(
                    onetile,
                    CopyTile<cb_x, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                    Abs<Dst::D0>{},
                    Negative<Dst::D0>{},
                    PackTile<cb_val, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#else
                eltwise_chain(
                    onetile,
                    CopyTile<cb_x, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                    Abs<Dst::D0>{},
                    PackTile<cb_val, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#endif
            }
#endif  // IS_ZERO

            // Add(x != 0)
            if (inner_idx == 0) {
                // PARTIAL migration: seed cb_cal with first cb_val tile.
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_val);
                pack_reconfig_data_format(cb_cal);
#endif
                {
                    using namespace compute_kernel_lib;
                    eltwise_chain(
                        onetile,
                        CopyTile<cb_val, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                        PackTile<cb_cal, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
            } else {
                tile_regs_acquire();
                cb_wait_front(cb_val, onetile);
                cb_wait_front(cb_cal, onetile);
                cb_reserve_back(cb_cal, onetile);
#ifdef IS_ZERO
                add_tiles_init_with_dt(cb_val, cb_cal);
                add_tiles(cb_val, cb_cal, 0, 0, dst0);
#else
                copy_tile_init_with_dt(cb_val);
                copy_tile(cb_val, 0, dst0);

                copy_tile_init_with_dt(cb_cal);
                copy_tile(cb_cal, 0, dst1);

                binary_max_tile_init();
                binary_max_tile(dst0, dst1, dst0);
#endif
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_cal);
                tile_regs_release();

                cb_pop_front(cb_val, onetile);
                cb_pop_front(cb_cal, onetile);
                cb_push_back(cb_cal, onetile);
            }
        }

        // PARTIAL migration: write cb_cal -> [Negative if MINUS_INF] -> cb_y.
#if defined FP32_DEST_ACC_EN
        reconfig_data_format_srca(cb_cal);
        pack_reconfig_data_format(cb_y);
#endif
        {
            using namespace compute_kernel_lib;
#ifdef MINUS_INF
            eltwise_chain(
                onetile,
                CopyTile<cb_cal, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                Negative<Dst::D0>{},
                PackTile<cb_y, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#else
            eltwise_chain(
                onetile,
                CopyTile<cb_cal, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                PackTile<cb_y, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#endif
        }
    }
    cb_pop_front(cb_one, onetile);
}
