// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    // WRITER RUNTIME ARGS
    uint32_t num_blocks = get_arg(args::num_blocks);
    uint32_t q_out_h_dim = get_arg(args::q_out_h_dim);
    uint32_t q_out_tensor_tile_id = get_arg(args::q_out_tensor_tile_id);
    uint32_t k_out_tensor_tile_id = get_arg(args::k_out_tensor_tile_id);
    uint32_t v_out_tensor_tile_id = get_arg(args::v_out_tensor_tile_id);

    // COMPILE TIME ARGS
    constexpr uint32_t q_out_h_tiles = get_arg(args::q_out_h_tiles);
    constexpr uint32_t q_out_w_tiles = get_arg(args::q_out_w_tiles);
    constexpr uint32_t q_out_HtWt = get_arg(args::q_out_HtWt);
    constexpr uint32_t q_out_c = get_arg(args::q_out_c);
    constexpr uint32_t kv_out_c = get_arg(args::kv_out_c);

    const auto sq = TensorAccessor(tensor::q);
    const auto sk = TensorAccessor(tensor::k);
    const auto sv = TensorAccessor(tensor::v);

    DataflowBuffer cb_qv(dfb::qv);  // cb for Q, V heads tiles
#ifdef TRANSPOSE_K_HEADS
    DataflowBuffer cb_k(dfb::k_out);  // cb for K heads (filled by compute)
#else
    DataflowBuffer cb_k(dfb::qv);  // cb for K heads (directly from reader; same buffer as Q, V)
#endif

    const uint32_t tile_bytes_qv = cb_qv.get_tile_size();
    const uint32_t tile_bytes_k = cb_k.get_tile_size();

    constexpr uint32_t block_size = 1;  // micro-block size for read/write; nothing to do with num_blocks
    // TODO: This might negatively impact perf
    constexpr uint32_t out_num_tiles_read = block_size;  // always read and pop by micro-block size for generality
    uint32_t l1_read_addr;
    uint32_t q_out_tensor_current_tile_id;  // need this to update q_out_tensor_tile_id
    uint32_t k_out_tensor_current_tile_id;  // need this to update k_out_tensor_tile_id
    uint32_t v_out_tensor_current_tile_id;  // need this to update v_out_tensor_tile_id
    uint32_t out_tensor_current_tile_id_along_c;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // q + create q head --> outputs: [B, num_q_heads, S, head_dim]
        out_tensor_current_tile_id_along_c = q_out_tensor_tile_id;
        for (uint32_t c_dim = 0; c_dim < q_out_c; c_dim++) {
            q_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                cb_qv.wait_front(out_num_tiles_read);
                l1_read_addr = cb_qv.get_read_ptr();
                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr),
                    sq,
                    tile_bytes_qv,
                    {},
                    {.page_id = q_out_tensor_current_tile_id});

                noc.async_write_barrier();
                cb_qv.pop_front(out_num_tiles_read);

                q_out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += q_out_HtWt;
        }

// k + create k head --> outputs: [B, num_kv_heads, S, head_dim]
#ifndef TRANSPOSE_K_HEADS
        out_tensor_current_tile_id_along_c = k_out_tensor_tile_id;
#else
        k_out_tensor_current_tile_id = k_out_tensor_tile_id;
#endif
        for (uint32_t c_dim = 0; c_dim < kv_out_c; c_dim++) {
#ifndef TRANSPOSE_K_HEADS
            k_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
#endif
            for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                cb_k.wait_front(out_num_tiles_read);
                l1_read_addr = cb_k.get_read_ptr();
                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr),
                    sk,
                    tile_bytes_k,
                    {},
                    {.page_id = k_out_tensor_current_tile_id});

                noc.async_write_barrier();
                cb_k.pop_front(out_num_tiles_read);

#ifndef TRANSPOSE_K_HEADS
                k_out_tensor_current_tile_id++;
#else
                k_out_tensor_current_tile_id += q_out_h_tiles;
#endif
            }
#ifndef TRANSPOSE_K_HEADS
            out_tensor_current_tile_id_along_c += q_out_HtWt;
#endif
        }

        // v + create v head --> outputs: [B, num_kv_heads, S, head_dim]
        out_tensor_current_tile_id_along_c = v_out_tensor_tile_id;
        for (uint32_t c_dim = 0; c_dim < kv_out_c; c_dim++) {
            v_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                cb_qv.wait_front(out_num_tiles_read);
                l1_read_addr = cb_qv.get_read_ptr();
                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr),
                    sv,
                    tile_bytes_qv,
                    {},
                    {.page_id = v_out_tensor_current_tile_id});

                noc.async_write_barrier();
                cb_qv.pop_front(out_num_tiles_read);

                v_out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += q_out_HtWt;
        }

        // Update out_tensor_tile_id for next h_dim or batch if we finish one CHtWt
        q_out_h_dim++;
        if (q_out_h_dim < q_out_h_tiles) {
            q_out_tensor_tile_id += q_out_w_tiles;
#ifndef TRANSPOSE_K_HEADS
            k_out_tensor_tile_id += q_out_w_tiles;
#else
            k_out_tensor_tile_id++;
#endif
            v_out_tensor_tile_id += q_out_w_tiles;
        } else {
            // If we finish one batch, always roll over to next tile in memory
            // This is just the current_tile_id, except for K when we transpose heads
            // In this case, decrement k_out_tensor_current_tile_id by the stride (q_out_h_tiles) and add 1 to roll over
            q_out_tensor_tile_id = q_out_tensor_current_tile_id;
#ifndef TRANSPOSE_K_HEADS
            k_out_tensor_tile_id = k_out_tensor_current_tile_id;
#else
            k_out_tensor_tile_id = ++k_out_tensor_current_tile_id - q_out_h_tiles;  // inc by 1 and decrement by stride
#endif
            v_out_tensor_tile_id = v_out_tensor_current_tile_id;
            q_out_h_dim = 0;
        }
    }
}
