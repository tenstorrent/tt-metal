// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <bool NeedsTypecast, uint32_t TcInFmt, uint32_t TcOutFmt>
ALWI void maybe_typecast_stat(
    DataflowBuffer& src_obj, uint32_t src_dfb, uint32_t dst_dfb, uint32_t& last_srca_dfb, uint32_t tile_index) {
    if constexpr (NeedsTypecast) {
        constexpr uint32_t onetile = 1;
        src_obj.wait_front(onetile);
        DataflowBuffer dst_obj(dst_dfb);
        dst_obj.reserve_back(onetile);

        tile_regs_acquire();
        reconfig_data_format_srca(last_srca_dfb, src_dfb);
        copy_init(src_dfb);
        last_srca_dfb = src_dfb;
        copy_tile(src_dfb, tile_index, tile_index * 2);
        typecast_tile_init<TcInFmt, TcOutFmt>();
        typecast_tile<TcInFmt, TcOutFmt>(tile_index * 2);
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(dst_dfb);
        pack_tile(tile_index * 2, dst_dfb);
        tile_regs_release();

        pack_reconfig_data_format(dst_dfb, src_dfb);

        src_obj.pop_front(onetile);
        dst_obj.push_back(onetile);
    }
}

// A writer-facing stat DFB is only bound when the accumulation format is wider than the stat dtype;
// on the other path the writer drains the compute output directly, so the same kernel-side handle
// has to name a different DFB. The aliases are gated at the preprocessor stage because
// dfb::writer_updated_* simply does not exist on the untypecast build. The host computes each flag
// (stat present AND the stat format needs a typecast) so one define gates one alias; the two stats
// are keyed independently, and either may typecast without the other.
#ifdef NEEDS_MEAN_TYPECAST
constexpr bool needs_mean_typecast = true;
constexpr auto dfb_writer_updated_mean = dfb::writer_updated_mean;
#else
constexpr bool needs_mean_typecast = false;
constexpr auto dfb_writer_updated_mean = dfb::updated_mean;
#endif

#ifdef NEEDS_VAR_TYPECAST
constexpr bool needs_var_typecast = true;
constexpr auto dfb_writer_updated_var = dfb::writer_updated_var;
#else
constexpr bool needs_var_typecast = false;
constexpr auto dfb_writer_updated_var = dfb::updated_var;
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t old_running_mean_has_value = get_arg(args::old_running_mean_has_value) == 1;
    constexpr uint32_t old_running_var_has_value = get_arg(args::old_running_var_has_value) == 1;
    static_assert(
        old_running_mean_has_value || old_running_var_has_value,
        "running_statistics requires at least one of running_mean / running_var");

    constexpr uint32_t tc_in_fmt = get_arg(args::tc_in_fmt);
    constexpr uint32_t tc_out_fmt = get_arg(args::tc_out_fmt);

    DataflowBuffer dfb_batch_mean_obj(dfb::batch_mean);
    DataflowBuffer dfb_batch_var_obj(dfb::batch_var);
    DataflowBuffer dfb_out0_obj(dfb::out);
    DataflowBuffer dfb_old_running_mean_obj(dfb::old_running_mean);
    DataflowBuffer dfb_old_running_var_obj(dfb::old_running_var);
    DataflowBuffer dfb_updated_running_mean_obj(dfb::updated_mean);
    DataflowBuffer dfb_updated_running_var_obj(dfb::updated_var);
    DataflowBuffer dfb_momentum_obj(dfb::momentum);
    DataflowBuffer dfb_one_obj(dfb::one);  // holds 1, for the (1 - momentum) term
    DataflowBuffer dfb_tmp1_obj(dfb::tmp1);
    DataflowBuffer dfb_tmp2_obj(dfb::tmp2);
    DataflowBuffer dfb_tmp3_obj(dfb::tmp3);

    compute_kernel_hw_startup(dfb::batch_mean, dfb::out);
    copy_init(dfb::batch_mean);
    uint32_t last_srca_dfb = dfb::batch_mean;
    constexpr uint32_t onetile = 1;

    dfb_momentum_obj.wait_front(1);
    dfb_one_obj.wait_front(1);

    // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // dfb tile index
        constexpr uint32_t tile_index = 0;

        dfb_batch_mean_obj.wait_front(onetile);
        dfb_batch_var_obj.wait_front(onetile);
        dfb_out0_obj.reserve_back(1);

        if constexpr (old_running_mean_has_value) {
            // 1 - momentum
            dfb_tmp1_obj.reserve_back(onetile);
            tile_regs_acquire();
            sub_binary_tile_init();
            reconfig_data_format_srca(last_srca_dfb, dfb::one);
            copy_init(dfb::one);
            last_srca_dfb = dfb::one;
            copy_tile(dfb::one, tile_index, tile_index * 2);
            reconfig_data_format_srca(last_srca_dfb, dfb::momentum);
            copy_init(dfb::momentum);
            last_srca_dfb = dfb::momentum;
            copy_tile(dfb::momentum, tile_index, tile_index * 2 + 1);
            sub_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(dfb::tmp1);
            pack_tile_with_dt(tile_index * 2, dfb::tmp1);
            tile_regs_release();
            dfb_tmp1_obj.push_back(onetile);

            // momentum * batch stat
            dfb_tmp2_obj.reserve_back(onetile);
            tile_regs_acquire();
            mul_binary_tile_init();
            reconfig_data_format_srca(last_srca_dfb, dfb::batch_mean);
            copy_init(dfb::batch_mean);
            last_srca_dfb = dfb::batch_mean;
            copy_tile(dfb::batch_mean, tile_index, tile_index * 2);
            reconfig_data_format_srca(last_srca_dfb, dfb::momentum);
            copy_init(dfb::momentum);
            last_srca_dfb = dfb::momentum;
            copy_tile(dfb::momentum, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            // No pack reconfig needed: tmp1 and tmp2 share interm_data_format
            pack_tile_with_dt(tile_index * 2, dfb::tmp2);
            tile_regs_release();
            dfb_tmp2_obj.push_back(onetile);

            // tmp1 * running stats --> (1 - momentum) * running stats
            dfb_tmp1_obj.wait_front(onetile);
            dfb_old_running_mean_obj.wait_front(onetile);
            dfb_tmp3_obj.reserve_back(onetile);
            tile_regs_acquire();
            reconfig_data_format_srca(last_srca_dfb, dfb::old_running_mean);
            copy_init(dfb::old_running_mean);
            last_srca_dfb = dfb::old_running_mean;
            copy_tile(dfb::old_running_mean, tile_index, tile_index * 2);
            reconfig_data_format_srca(last_srca_dfb, dfb::tmp1);
            copy_init(dfb::tmp1);
            last_srca_dfb = dfb::tmp1;
            copy_tile(dfb::tmp1, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            // No pack reconfig needed: tmp2 and tmp3 share interm_data_format
            pack_tile_with_dt(tile_index * 2, dfb::tmp3);
            tile_regs_release();
            dfb_tmp3_obj.push_back(onetile);

            dfb_old_running_mean_obj.pop_front(onetile);
            dfb_tmp1_obj.pop_front(onetile);

            // tmp2 + tmp3 --> (momentum * batch stat) + ((1 - momentum) * running stats)
            dfb_tmp2_obj.wait_front(onetile);
            dfb_tmp3_obj.wait_front(onetile);
            dfb_updated_running_mean_obj.reserve_back(onetile);
            tile_regs_acquire();
            add_binary_tile_init();
            reconfig_data_format_srca(last_srca_dfb, dfb::tmp3);
            copy_init(dfb::tmp3);
            last_srca_dfb = dfb::tmp3;
            copy_tile(dfb::tmp3, tile_index, tile_index * 2);
            reconfig_data_format_srca(last_srca_dfb, dfb::tmp2);
            copy_init(dfb::tmp2);
            last_srca_dfb = dfb::tmp2;
            copy_tile(dfb::tmp2, tile_index, tile_index * 2 + 1);
            add_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            // No pack reconfig needed: tmp3 and updated_mean share interm_data_format
            pack_tile_with_dt(tile_index * 2, dfb::updated_mean);
            // For the output tensor, return the same values as either of the stats.
            if constexpr (!old_running_var_has_value) {
                pack_reconfig_data_format(dfb::updated_mean, dfb::out);
                pack_tile_with_dt(tile_index * 2, dfb::out);
            }
            tile_regs_release();
            dfb_updated_running_mean_obj.push_back(onetile);

            maybe_typecast_stat<needs_mean_typecast, tc_in_fmt, tc_out_fmt>(
                dfb_updated_running_mean_obj, dfb::updated_mean, dfb_writer_updated_mean, last_srca_dfb, tile_index);

            dfb_tmp3_obj.pop_front(onetile);
            dfb_tmp2_obj.pop_front(onetile);
        }

        if constexpr (old_running_var_has_value) {
            // 1 - momentum
            dfb_tmp1_obj.reserve_back(onetile);
            tile_regs_acquire();
            sub_binary_tile_init();
            reconfig_data_format_srca(last_srca_dfb, dfb::one);
            copy_init(dfb::one);
            last_srca_dfb = dfb::one;
            copy_tile(dfb::one, tile_index, tile_index * 2);
            reconfig_data_format_srca(last_srca_dfb, dfb::momentum);
            copy_init(dfb::momentum);
            last_srca_dfb = dfb::momentum;
            copy_tile(dfb::momentum, tile_index, tile_index * 2 + 1);
            sub_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(dfb::tmp1);
            pack_tile_with_dt(tile_index * 2, dfb::tmp1);
            tile_regs_release();
            dfb_tmp1_obj.push_back(onetile);

            // momentum * batch stat
            dfb_tmp2_obj.reserve_back(onetile);
            tile_regs_acquire();
            mul_binary_tile_init();
            reconfig_data_format_srca(last_srca_dfb, dfb::batch_var);
            copy_init(dfb::batch_var);
            last_srca_dfb = dfb::batch_var;
            copy_tile(dfb::batch_var, tile_index, tile_index * 2);
            reconfig_data_format_srca(last_srca_dfb, dfb::momentum);
            copy_init(dfb::momentum);
            last_srca_dfb = dfb::momentum;
            copy_tile(dfb::momentum, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, dfb::tmp2);
            tile_regs_release();
            dfb_tmp2_obj.push_back(onetile);

            // tmp1 * running stats --> (1 - momentum) * running stats
            dfb_tmp1_obj.wait_front(onetile);
            dfb_old_running_var_obj.wait_front(onetile);
            dfb_tmp3_obj.reserve_back(onetile);
            tile_regs_acquire();
            reconfig_data_format_srca(last_srca_dfb, dfb::old_running_var);
            copy_init(dfb::old_running_var);
            last_srca_dfb = dfb::old_running_var;
            copy_tile(dfb::old_running_var, tile_index, tile_index * 2);
            reconfig_data_format_srca(last_srca_dfb, dfb::tmp1);
            copy_init(dfb::tmp1);
            last_srca_dfb = dfb::tmp1;
            copy_tile(dfb::tmp1, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, dfb::tmp3);
            tile_regs_release();
            dfb_tmp3_obj.push_back(onetile);

            dfb_old_running_var_obj.pop_front(onetile);
            dfb_tmp1_obj.pop_front(onetile);

            // tmp2 + tmp3 --> (momentum * batch stat) + ((1 - momentum) * running stats)
            dfb_tmp2_obj.wait_front(onetile);
            dfb_tmp3_obj.wait_front(onetile);
            dfb_updated_running_var_obj.reserve_back(onetile);
            tile_regs_acquire();
            add_binary_tile_init();
            reconfig_data_format_srca(last_srca_dfb, dfb::tmp3);
            copy_init(dfb::tmp3);
            last_srca_dfb = dfb::tmp3;
            copy_tile(dfb::tmp3, tile_index, tile_index * 2);
            reconfig_data_format_srca(last_srca_dfb, dfb::tmp2);
            copy_init(dfb::tmp2);
            last_srca_dfb = dfb::tmp2;
            copy_tile(dfb::tmp2, tile_index, tile_index * 2 + 1);
            add_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, dfb::updated_var);
            pack_reconfig_data_format(dfb::updated_var, dfb::out);
            pack_tile_with_dt(tile_index * 2, dfb::out);
            tile_regs_release();
            dfb_updated_running_var_obj.push_back(onetile);

            maybe_typecast_stat<needs_var_typecast, tc_in_fmt, tc_out_fmt>(
                dfb_updated_running_var_obj, dfb::updated_var, dfb_writer_updated_var, last_srca_dfb, tile_index);

            dfb_tmp3_obj.pop_front(onetile);
            dfb_tmp2_obj.pop_front(onetile);
        }

        dfb_batch_mean_obj.pop_front(onetile);
        dfb_batch_var_obj.pop_front(onetile);
        dfb_out0_obj.push_back(1);
    }
    dfb_momentum_obj.pop_front(1);
    dfb_one_obj.pop_front(1);
}
