// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 compute kernel for the multi-core W reduction primitive *with negation*.
//
// Migration notes:
//   - Compile-time arguments are bound by name (`args::Wt`, `args::NC`,
//     `args::post_mul_scaler_bits`).
//   - `Ht` is bound as a per-node *runtime* argument (`args::Ht`); see the comment in
//     `reduce.cpp` for why.
//   - Local DataflowBuffers are bound by name (`dfb::input`, `dfb::scaler`,
//     `dfb::output`, `dfb::acc_w`/`dfb::acc_r`, `dfb::ineg_w`/`dfb::ineg_r`).
//   - The accumulator and intermediate-negation buffers are produced AND consumed by
//     this same compute kernel. Metal 2.0 requires the producer and consumer DFBBindings
//     to have distinct `local_accessor_name`s on the same kernel, so each of these two
//     DFBs has two host-side bindings (`*_w` for PRODUCER, `*_r` for CONSUMER). On
//     Gen1 the resulting DFBAccessor ids both resolve to the same underlying CB. On
//     Gen2 they resolve to the same underlying DFB; the writer/reader views just
//     express the producer/consumer roles to the host validator.
//   - All sync (wait/pop/reserve/push) is done through the typed DataflowBuffer
//     wrapper, which works on both Gen1 and Gen2 with identical syntax. The LLK
//     compute calls (copy_tile, reduce_tile, pack_tile, ...) still take raw uint32_t
//     ids; we get those from each buffer's `.get_id()`.

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

    // Typed dataflow-buffer wrappers. On Gen1 these forward to circular_buffer_interface
    // ops; on Gen2 they drive real DFB hardware. The same source compiles on both.
    experimental::DataflowBuffer input_buf(dfb::input);
    experimental::DataflowBuffer scaler_buf(dfb::scaler);
    experimental::DataflowBuffer output_buf(dfb::output);
    experimental::DataflowBuffer acc_writer(dfb::acc_w);
    experimental::DataflowBuffer acc_reader(dfb::acc_r);
    experimental::DataflowBuffer ineg_writer(dfb::ineg_w);
    experimental::DataflowBuffer ineg_reader(dfb::ineg_r);

    // LLK calls (copy_tile, reduce_init, reduce_tile, pack_tile, ...) still take
    // raw buffer ids. Pull them from the typed wrappers once, up front.
    const uint32_t input_id = input_buf.get_id();
    const uint32_t scaler_id = scaler_buf.get_id();
    const uint32_t output_id = output_buf.get_id();
    const uint32_t acc_id = acc_writer.get_id();    // == acc_reader.get_id()
    const uint32_t ineg_id = ineg_writer.get_id();  // == ineg_reader.get_id()

    compute_kernel_hw_startup(input_id, scaler_id, output_id);

    scaler_buf.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                input_buf.wait_front(onetile);
                tile_regs_acquire();
                copy_tile_init(input_id);
                copy_tile(input_id, 0, dst_idx);
                negative_tile_init();
                negative_tile(dst_idx);
                tile_regs_wait();
                input_buf.pop_front(onetile);
                ineg_writer.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, ineg_id);
                tile_regs_release();
                ineg_writer.push_back(onetile);

                tile_regs_acquire();
                if (wt > 0) {
                    acc_reader.wait_front(onetile);
                    copy_tile_init(acc_id);
                    copy_tile(acc_id, 0, dst_idx);
                }

                ineg_reader.wait_front(onetile);
                reduce_init<REDUCE_OP, REDUCE_DIM>(ineg_id, scaler_id, acc_id);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(ineg_id, scaler_id, 0, 0, dst_idx);
                reduce_uninit();
                tile_regs_wait();
                ineg_reader.pop_front(onetile);
                if (wt > 0) {
                    acc_reader.pop_front(onetile);
                }
                acc_writer.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, acc_id);
                tile_regs_release();
                acc_writer.push_back(onetile);
            }  // wt

            acc_reader.wait_front(onetile);
            tile_regs_acquire();
            copy_tile_init(acc_id);
            copy_tile(acc_id, 0, dst_idx);
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
            acc_reader.pop_front(onetile);
            output_buf.reserve_back(onetile);
            tile_regs_commit();
            pack_tile(dst_idx, output_id);
            tile_regs_release();
            output_buf.push_back(onetile);
        }  // ht
    }  // nc
}
