// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#ifdef IS_ZERO
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.hpp"
#endif
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);

    constexpr std::uint32_t cb_x = tt::CB::c_in0 + 0;       // input
    constexpr std::uint32_t cb_one = tt::CB::c_in0 + 1;     // one
    constexpr std::uint32_t cb_mask_h = tt::CB::c_in0 + 2;  // mask_h

    constexpr std::uint32_t cb_y = tt::CB::c_out0 + 0;  // output

    constexpr std::uint32_t cb_tmp0 = tt::CB::c_intermed0 + 0;
    constexpr std::uint32_t cb_tmp1 = tt::CB::c_intermed0 + 1;
    constexpr std::uint32_t cb_tmp2 = tt::CB::c_intermed0 + 2;

    constexpr std::uint32_t cb_val = cb_tmp0;     // f(x)
    constexpr std::uint32_t cb_cal = cb_tmp1;     // calculate f(x) over dimension
    constexpr std::uint32_t cb_reduce = cb_tmp2;  // reduce f(x)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    cb_wait_front(cb_one, onetile);  // comes from the reader

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    if (do_mask_h) {
        cb_wait_front(cb_mask_h, onetile);  // comes from the reader
    }
    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            // f(x)
#ifdef MINUS_INF
            // BLOCKED: mask_posinf_tile + negative_tile pattern. No chain op for
            // mask_posinf — leave raw.
            tile_regs_acquire();
            cb_wait_front(cb_x, onetile);  // comes from the reader
            cb_reserve_back(cb_val, onetile);

            copy_tile_init_with_dt(cb_x);
            copy_tile(cb_x, 0, dst0);

            if (do_mask_h && (row_idx == Ht - 1)) {
                copy_tile_init_with_dt(cb_mask_h);
                copy_tile(cb_mask_h, 0, dst1);

                mask_tile_init();
                mask_posinf_tile(dst0, dst1);
            }
#ifdef IS_ZERO
            unary_ne_tile_init();
            unary_ne_tile(dst0, 0);
#else
            abs_tile_init();
            abs_tile(dst0);
#endif

            negative_tile_init();
            negative_tile(dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_val);
            tile_regs_release();

            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_val, onetile);
#else
            // PARTIAL migration: f(x) prologue (plain mask_tile path) via eltwise_chain.
            // The masked branch loads cb_x + cb_mask_h into D0/D1 (cb_mask_h is
            // pre-waited outside the loop, hence NoWaitNoPop), applies Mask, then the
            // per-macro SFPU (UnaryNe under IS_ZERO, else Abs).
#if defined FP32_DEST_ACC_EN
            reconfig_data_format_srca(cb_x);
            pack_reconfig_data_format(cb_val);
#endif
            if (do_mask_h && (row_idx == Ht - 1)) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::
                        CopyTile<cb_x, compute_kernel_lib::Dst::D0, compute_kernel_lib::CopyTilePolicy::WaitAndPop>{},
                    compute_kernel_lib::CopyTile<
                        cb_mask_h,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::CopyTilePolicy::NoWaitNoPop>{},
                    compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
#ifdef IS_ZERO
                    compute_kernel_lib::UnaryNe<compute_kernel_lib::Dst::D0>{0u},
#else
                    compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
#endif
                    compute_kernel_lib::PackTile<
                        cb_val,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush>{});
            } else {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::
                        CopyTile<cb_x, compute_kernel_lib::Dst::D0, compute_kernel_lib::CopyTilePolicy::WaitAndPop>{},
#ifdef IS_ZERO
                    compute_kernel_lib::UnaryNe<compute_kernel_lib::Dst::D0>{0u},
#else
                    compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
#endif
                    compute_kernel_lib::PackTile<
                        cb_val,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush>{});
            }
#endif  // MINUS_INF

            // calculate f(x) over dimension
            if (row_idx == 0) {
                // PARTIAL migration: copy cb_val -> cb_cal as init seed.
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
#ifdef IS_ZERO
                // PARTIAL migration: in-place accumulator add via eltwise_chain.
                // cb_cal = cb_val + cb_cal
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        cb_val,
                        cb_cal,
                        cb_cal,
                        compute_kernel_lib::BinaryFpuOp::Add,
                        compute_kernel_lib::BroadcastDim::None,
                        compute_kernel_lib::BinaryDataFormatReconfig::InputAndOutput,
                        compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                        compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                        compute_kernel_lib::CbIndexMode::FirstTile,
                        compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<
                        cb_cal,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush>{});
#else
                // BLOCKED: binary_max_tile is a held-DEST SFPU op pattern
                // (copy two tiles into DEST, then max in-place). No BinaryMax
                // op struct in eltwise_chain — leave on raw LLK.
                tile_regs_acquire();
                cb_wait_front(cb_val, onetile);
                cb_wait_front(cb_cal, onetile);
                cb_reserve_back(cb_cal, onetile);
                copy_tile_init_with_dt(cb_val);
                copy_tile(cb_val, 0, dst0);

                copy_tile_init_with_dt(cb_cal);
                copy_tile(cb_cal, 0, dst1);

                binary_max_tile_init();
                binary_max_tile(dst0, dst1, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_cal);
                tile_regs_release();

                cb_pop_front(cb_val, onetile);
                cb_pop_front(cb_cal, onetile);
                cb_push_back(cb_cal, onetile);
#endif
            }
        }
        // reduce f(x)
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
            cb_cal, cb_one, cb_reduce, compute_kernel_lib::ReduceInputBlockShape::single());

        // PARTIAL migration: post-reduce write-out (cb_reduce -> [Negative if MINUS_INF] -> cb_y).
#if defined FP32_DEST_ACC_EN
        reconfig_data_format_srca(cb_reduce);
        pack_reconfig_data_format(cb_y);
#endif
        {
            using namespace compute_kernel_lib;
#ifdef MINUS_INF
            eltwise_chain(
                onetile,
                CopyTile<cb_reduce, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                Negative<Dst::D0>{},
                PackTile<cb_y, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#else
            eltwise_chain(
                onetile,
                CopyTile<cb_reduce, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                PackTile<cb_y, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#endif
        }
    }

    cb_pop_front(cb_one, onetile);
    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
}
