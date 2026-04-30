// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 compute kernel for the multi-core W reduction primitive *with negation*.
//
// Migration notes:
//   - Compile-time arguments are bound by name (`args::Wt`, `args::NC`,
//     `args::post_mul_scaler_bits`).
//   - `Ht` is bound as a per-node *runtime* argument (`args::Ht`); see the comment in
//     `reduce_metal2.cpp` for why.
//   - Local DataflowBuffers are bound by name (`dfb::input`, `dfb::scaler`,
//     `dfb::output`, `dfb::acc_w`/`dfb::acc_r`, `dfb::ineg_w`/`dfb::ineg_r`).
//   - The accumulator and intermediate-negation buffers are produced AND consumed by
//     this same compute kernel. Metal 2.0 requires the producer and consumer DFBBindings
//     to have distinct `local_accessor_name`s on the same kernel, so each of these two
//     DFBs has two host-side bindings (`*_w` for PRODUCER, `*_r` for CONSUMER). On
//     Gen1 the resulting DFBAccessor ids both resolve to the same underlying CB, so
//     in this kernel we use the writer view for reserve/push and the reader view for
//     wait/pop on the same physical buffer.

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/dataflow_buffer.h"

#include "llk_math_eltwise_binary.h"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    const uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);
#ifdef REDUCE_POST_MUL
    // Packed fp32 user scalar applied via mul_unary_tile after the reduce+negate finishes.
    constexpr uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif

    // DFB accessor ids (constexpr, used by the LLK helpers that take CB ids by value).
    constexpr uint32_t cb_input = dfb::input.id;
    constexpr uint32_t cb_scaler = dfb::scaler.id;
    constexpr uint32_t cb_output = dfb::output.id;
    constexpr uint32_t cb_acc = dfb::acc_w.id;    // == dfb::acc_r.id at runtime
    constexpr uint32_t cb_ineg = dfb::ineg_w.id;  // == dfb::ineg_r.id at runtime

    // DFB objects (writer/reader views share the same underlying CB on Gen1).
    experimental::DataflowBuffer cb_input_obj(dfb::input);
    experimental::DataflowBuffer cb_scaler_obj(dfb::scaler);
    experimental::DataflowBuffer cb_output_obj(dfb::output);
    experimental::DataflowBuffer cb_acc_writer(dfb::acc_w);
    experimental::DataflowBuffer cb_acc_reader(dfb::acc_r);
    experimental::DataflowBuffer cb_ineg_writer(dfb::ineg_w);
    experimental::DataflowBuffer cb_ineg_reader(dfb::ineg_r);

    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);

    cb_scaler_obj.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_input_obj.wait_front(onetile);
                tile_regs_acquire();
                copy_tile_init(cb_input);
                copy_tile(cb_input, 0, dst_idx);
                negative_tile_init();
                negative_tile(dst_idx);
                tile_regs_wait();
                cb_input_obj.pop_front(onetile);
                cb_ineg_writer.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, cb_ineg);
                tile_regs_release();
                cb_ineg_writer.push_back(onetile);

                tile_regs_acquire();
                if (wt > 0) {
                    cb_acc_reader.wait_front(onetile);
                    copy_tile_init(cb_acc);
                    copy_tile(cb_acc, 0, dst_idx);
                }

                cb_ineg_reader.wait_front(onetile);
                reduce_init<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, cb_acc);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, 0, 0, dst_idx);
                reduce_uninit();
                tile_regs_wait();
                cb_ineg_reader.pop_front(onetile);
                if (wt > 0) {
                    cb_acc_reader.pop_front(onetile);
                }
                cb_acc_writer.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, cb_acc);
                tile_regs_release();
                cb_acc_writer.push_back(onetile);
            }  // wt

            cb_acc_reader.wait_front(onetile);
            tile_regs_acquire();
            copy_tile_init(cb_acc);
            copy_tile(cb_acc, 0, dst_idx);
            negative_tile_init();
            negative_tile(dst_idx);
#ifdef REDUCE_POST_MUL
            // GMPOOL only respects the scaler's exponent for MAX/MIN, so the host requests reduction
            // with scaler=1.0 and then applies the user scalar via mul_unary_tile (SFPU) on each
            // output DEST register.
            binop_with_scalar_tile_init();
            mul_unary_tile(dst_idx, post_mul_scaler_bits);
#endif
            tile_regs_wait();
            cb_acc_reader.pop_front(onetile);
            cb_output_obj.reserve_back(onetile);
            tile_regs_commit();
            pack_tile(dst_idx, cb_output);
            tile_regs_release();
            cb_output_obj.push_back(onetile);
        }  // ht
    }  // nc
}
