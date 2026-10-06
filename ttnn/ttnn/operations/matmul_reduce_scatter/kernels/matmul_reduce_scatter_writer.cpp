// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — compute-core BRISC: weight (W, in1) operand.
//
// Per scatter block (in compute order) and per K-block: the n-line injector (W_SENDS) reads its columns' K-block
// from DRAM and multicasts it along its n-line (mcast_pipe SenderPipe); the other cores of the line receive it.
// When W is the block-invariant operand (scatter_dim=-2) and resident (regime R1), blocks > 0 only replay CB
// credits (capacity = one K pass exactly, so the ring wraps onto the same pages).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_weight_operand = get_compile_time_arg_val(0);
    constexpr uint32_t core_n_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t w_sends = get_compile_time_arg_val(6);
    constexpr uint32_t w_resident = get_compile_time_arg_val(7);
    constexpr auto w_args = TensorAccessorArgs<8>();
    constexpr auto mc_w =
        McastArgs<get_named_compile_time_arg_val("w_ct_offset"), get_named_compile_time_arg_val("w_rt_offset")>();

    constexpr uint32_t kblock_pages = k_block_tiles * core_n_tiles;
    constexpr uint32_t kblock_bytes = kblock_pages * w_tile_bytes;

    size_t arg = 0;
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_row_stride = get_arg_val<uint32_t>(arg++);      // Nt (W pages per tile-row)
    const uint32_t w_col0 = get_arg_val<uint32_t>(arg++);            // n-line's first column within a block
    const uint32_t w_valid_cols = get_arg_val<uint32_t>(arg++);      // <= core_n_tiles (ragged last n-line)
    const uint32_t w_block_col_step = get_arg_val<uint32_t>(arg++);  // blk_n_tiles for scatter_dim=-1, else 0
    const uint32_t order_idx = arg;                                  // num_blocks x block j
    arg += num_blocks;

    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);

    Noc noc;
    // transfer(dst, col_base, kb): deliver one K-block of W into this core's CB at dst
    auto run = [&](auto&& transfer) {
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t col_base = get_arg_val<uint32_t>(order_idx + b) * w_block_col_step + w_col0;
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                cb_reserve_back(cb_weight_operand, kblock_pages);
                if (!(w_resident && b > 0)) {
                    transfer(get_write_ptr(cb_weight_operand), col_base, kb);
                }
                cb_push_back(cb_weight_operand, kblock_pages);
            }
        }
    };
    if constexpr (w_sends) {
        auto pipe = mc_w.sender(noc);
        run([&](uint32_t dst, uint32_t col_base, uint32_t kb) {
            // CB layout per K-block: [k_block_tiles][core_n_tiles]; padding columns stay unread.
            for (uint32_t k = 0; k < k_block_tiles; ++k) {
                const uint32_t page0 = (kb * k_block_tiles + k) * w_row_stride + col_base;
                for (uint32_t c = 0; c < w_valid_cols; ++c) {
                    noc_async_read(
                        w_acc.get_noc_addr(page0 + c), dst + (k * core_n_tiles + c) * w_tile_bytes, w_tile_bytes);
                }
            }
            noc_async_read_barrier();
            pipe.send(dst, dst, kblock_bytes);
        });
    } else {
        auto pipe = mc_w.receiver(noc);
        run([&](uint32_t, uint32_t, uint32_t) { pipe.receive(); });
    }
}
