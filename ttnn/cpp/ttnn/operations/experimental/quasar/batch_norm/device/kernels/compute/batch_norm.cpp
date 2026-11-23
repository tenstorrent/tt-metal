// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar batch_norm (inference), per-plane streamed compute.
//
// Threading: the kernel runs on C NEOs (num_threads = C). The unit of work is one (n, c) plane of HtWt
// tiles; NEO t owns planes t, t+C, ... of the core's num_planes. Every NEO runs rounds = ceil(num_planes / C)
// rounds; a NEO that runs out of planes recomputes NEO 0's plane for that round (the reader feeds it the
// same tiles and the writer stores the identical result to the same pages), so all NEOs, reader harts and
// writer harts stay in lock-step on the STRIDED rings.
//
// Per plane (channel c): the reader delivers one tile each of mean[c], var[c] and, when fused, weight[c] /
// bias[c]; the parameter value sits at element (0, 0) of its tile. The NEO folds them into two scalar tiles,
//     scale = weight * rsqrt(var + eps)          shift = bias - mean * scale
// (element (0, 0) is all that matters: the per-tile ops are scalar broadcasts of (0, 0)), then streams the
// plane as out = x * scale + shift, block_size tiles per DST acquire.
//
// Ring access: the rings shared with other NEOs are STRIDED (stride C) and the binary unpack indexes
// entries, not strides, so every binary op reads tile index 0 with a wait(1) / op / pop(1) rhythm.
//
// Reader contract (dfb::in):          per plane, tiles 0..HtWt-1.
// Reader contract (dfb::eps):         one bf16 scalar tile of eps per NEO, once per kernel.
// Reader contract (dfb::mean / var / weight / bias): one tile per plane.
// Writer contract (dfb::out):         per plane, tiles 0..HtWt-1 in order.
//
// Compile-time args: HtWt, block_size, compute_threads   Runtime args: num_planes
// Defines:           FUSE_WEIGHT, FUSE_BIAS (gate the dfb::weight / dfb::bias bindings)
//
// The pack destination is baked at pack_init; every pack-target switch re-inits.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_planes = get_arg(args::num_planes);
    constexpr uint32_t HtWt = get_arg(args::HtWt);
    constexpr uint32_t blk = get_arg(args::block_size);
    constexpr uint32_t C = get_arg(args::compute_threads);

    static_assert(HtWt > 0 && blk > 0 && C > 0, "bad geometry");
    static_assert(blk <= 4, "block is limited by DST capacity (fp32 acc)");

    constexpr auto dfb_in = dfb::in;
    constexpr auto dfb_eps = dfb::eps;
    constexpr auto dfb_mean = dfb::mean;
    constexpr auto dfb_var = dfb::var;
    constexpr auto dfb_rstd = dfb::rstd;
    constexpr auto dfb_scale = dfb::scale;
    constexpr auto dfb_tmp = dfb::tmp;
    constexpr auto dfb_shift = dfb::shift;
    constexpr auto dfb_xmm = dfb::xmm;
    constexpr auto dfb_out = dfb::out;
#ifdef FUSE_WEIGHT
    constexpr bool do_weight = true;
    constexpr auto dfb_weight = dfb::weight;
#else
    constexpr bool do_weight = false;
    constexpr auto dfb_weight = dfb_in;  // unused
#endif
#ifdef FUSE_BIAS
    constexpr bool do_bias = true;
    constexpr auto dfb_bias = dfb::bias;
#else
    constexpr bool do_bias = false;
    constexpr auto dfb_bias = dfb_in;  // unused
#endif

    // One object per buffer for the whole kernel (the drains run in ~DataflowBuffer, so no locals).
    DataflowBuffer in(dfb_in);
    DataflowBuffer eps(dfb_eps);
    DataflowBuffer mean(dfb_mean);
    DataflowBuffer var(dfb_var);
    DataflowBuffer rstd(dfb_rstd);
    DataflowBuffer scale(dfb_scale);
    DataflowBuffer tmp(dfb_tmp);
    DataflowBuffer shift(dfb_shift);
    DataflowBuffer xmm(dfb_xmm);
    DataflowBuffer out(dfb_out);
    DataflowBuffer weight(dfb_weight);
    DataflowBuffer bias(dfb_bias);

    constexpr uint32_t dst0 = 0;
    const uint32_t rounds = (num_planes + C - 1) / C;

    compute_kernel_hw_startup(dfb_in, dfb_mean, dfb_out);
    eps.wait_front(1);  // pushed once by the reader, popped after its last use

    // Pack `nb` DST tiles (0..nb-1) into `dst_dfb` as one batch.
    // `auto&`: DataflowBuffer is a class template (one type per access-pattern pair).
    auto pack_batch = [&](uint32_t dst_dfb, auto& dst_obj, uint32_t nb) __attribute__((always_inline)) {
        const auto nb16 = static_cast<uint16_t>(nb);
        dst_obj.reserve_back(nb16);
        tile_regs_wait();
        pack_reconfig_data_format(dst_dfb);
        pack_init(dst_dfb);  // BFD destination is baked at pack_init, not by the format reconfig
        for (uint32_t i = 0; i < nb; ++i) {
            pack_tile(i, dst_dfb);
        }
        tile_regs_release();
        dst_obj.push_back(nb16);
    };

    for (uint32_t round = 0; round < rounds; ++round) {
        // ---------------------------------------------------------- per-channel scale / shift
        // rstd = rsqrt(var + eps)
        tile_regs_acquire();
        reconfig_data_format(dfb_var, dfb_eps);
        add_init(dfb_var, dfb_eps);
        var.wait_front(1);
        add_tiles(dfb_var, dfb_eps, 0, 0, dst0);
        var.pop_front(1);
        if (round == rounds - 1) {
            eps.pop_front(1);  // last use: release it so the reader's drain of that ring can finish
        }
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();
        if constexpr (do_weight) {
            pack_batch(dfb_rstd, rstd, 1);
            // scale = rstd * weight
            tile_regs_acquire();
            reconfig_data_format(dfb_rstd, dfb_weight);
            mul_init(dfb_rstd, dfb_weight);
            rstd.wait_front(1);
            weight.wait_front(1);
            mul_tiles(dfb_rstd, dfb_weight, 0, 0, dst0);
            rstd.pop_front(1);
            weight.pop_front(1);
            tile_regs_commit();
        }
        pack_batch(dfb_scale, scale, 1);
        scale.wait_front(1);

        // shift = bias - mean * scale   (or -(mean * scale) without bias)
        tile_regs_acquire();
        reconfig_data_format(dfb_mean, dfb_scale);
        mul_init(dfb_mean, dfb_scale);
        mean.wait_front(1);
        mul_tiles(dfb_mean, dfb_scale, 0, 0, dst0);
        mean.pop_front(1);
        if constexpr (do_bias) {
            tile_regs_commit();
            pack_batch(dfb_tmp, tmp, 1);
            tile_regs_acquire();
            reconfig_data_format(dfb_bias, dfb_tmp);
            sub_init(dfb_bias, dfb_tmp);
            bias.wait_front(1);
            tmp.wait_front(1);
            sub_tiles(dfb_bias, dfb_tmp, 0, 0, dst0);
            bias.pop_front(1);
            tmp.pop_front(1);
            tile_regs_commit();
        } else {
            negative_tile_init();
            negative_tile(dst0);
            tile_regs_commit();
        }
        pack_batch(dfb_shift, shift, 1);
        shift.wait_front(1);

        // ---------------------------------------------------------- out = x * scale + shift
        for (uint32_t p = 0; p < HtWt; p += blk) {
            const uint32_t nb = (HtWt - p) < blk ? (HtWt - p) : blk;

            tile_regs_acquire();
            reconfig_data_format(dfb_in, dfb_scale);
            mul_bcast_scalar_init(dfb_in, dfb_scale);
            for (uint32_t i = 0; i < nb; ++i) {
                in.wait_front(1);
                mul_tiles_bcast_scalar(dfb_in, dfb_scale, 0, 0, i);
                in.pop_front(1);
            }
            tile_regs_commit();
            pack_batch(dfb_xmm, xmm, nb);

            tile_regs_acquire();
            reconfig_data_format(dfb_xmm, dfb_shift);
            add_bcast_scalar_init(dfb_xmm, dfb_shift);
            for (uint32_t i = 0; i < nb; ++i) {
                xmm.wait_front(1);
                add_tiles_bcast_scalar(dfb_xmm, dfb_shift, 0, 0, i);
                xmm.pop_front(1);
            }
            tile_regs_commit();
            pack_batch(dfb_out, out, nb);
        }

        scale.pop_front(1);
        shift.pop_front(1);
    }
}
