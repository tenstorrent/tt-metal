// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_to_all_matmul compute-core reader (RISCV_1 / NoC0).
//
// Per head slot of this core, in slot order (local-source heads first, then remote heads by source-local index j):
//   load_weight_block(slot, 0)   Kt x block_n weight tiles -> cb_weight (issued before the slot lands: the dominant
//                                DRAM read fills the fabric wait)
//   land_block(slot)             cb_a_landed is backed on the persistent landing scratch (zero-copy): slot s sits at
//                                base + s * slot_bytes, where the remote port writes it.
//                                local head:  read the own quarter of the head's rows (all batches, all K) from DRAM
//                                remote head: wait for the slot's arrival counter (fused inc on the entry's last
//                                             packet), re-arm it (-1)
//                                then push the slot (compute retains it across the head's n-blocks)
//   load_weight_block(slot, nb)  the head's remaining n-blocks
//
// Slot tile order (b, mt, k) = matmul_block's in0 order (rows of Kt tiles).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_a_landed = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t block_n = get_compile_time_arg_val(4);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t B = get_compile_time_arg_val(6);
    constexpr uint32_t Mq_t = get_compile_time_arg_val(7);
    constexpr uint32_t St = get_compile_time_arg_val(8);
    constexpr uint32_t Hl = get_compile_time_arg_val(9);
    constexpr uint32_t in_tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(11);
    constexpr auto in_args = TensorAccessorArgs<12>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t quarter_tiles = Mq_t * Kt;       // one batch's rows of one head quarter
    constexpr uint32_t slot_tiles = B * quarter_tiles;  // = block_m * Kt
    constexpr uint32_t weight_block_tiles = Kt * block_n;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t my_q = get_common_arg_val<uint32_t>(2);  // this chip's position in its group
    constexpr uint32_t arrival_base = 3;                    // arrival counter address of slot s

    const uint32_t n_slots = get_arg_val<uint32_t>(0);

    const auto in = TensorAccessor(in_args, in_addr, in_tile_bytes);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);

    auto load_weight_block = [&](uint32_t h, uint32_t nb) {
        cb_reserve_back(cb_weight, weight_block_tiles);
        uint32_t wp = get_write_ptr(cb_weight);
        uint32_t page = h * Kt * Nt + nb * block_n;
        for (uint32_t k = 0; k < Kt; ++k) {
            for (uint32_t n = 0; n < block_n; ++n) {
                noc_async_read(w.get_noc_addr(page + n), wp, w_tile_bytes);
                wp += w_tile_bytes;
            }
            page += Nt;
        }
        noc_async_read_barrier();
        cb_push_back(cb_weight, weight_block_tiles);
    };

    for (uint32_t s = 0; s < n_slots; ++s) {
        const uint32_t word = get_arg_val<uint32_t>(1 + s);
        const uint32_t h = word & 0xFFFF;
        const bool remote = (word >> 16) != 0;

        load_weight_block(h, 0);

        cb_reserve_back(cb_a_landed, slot_tiles);
        if (remote) {
            volatile tt_l1_ptr uint32_t* arrived =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(arrival_base + s));
            noc_semaphore_wait_min(arrived, 1);
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrived)), 0u - 1u);
            noc_async_atomic_barrier();
        } else {
            const uint32_t j = h - my_q * Hl;
            uint32_t ap = get_write_ptr(cb_a_landed);
            for (uint32_t b = 0; b < B; ++b) {
                const uint32_t base = ((b * Hl + j) * St + my_q * Mq_t) * Kt;
                for (uint32_t r = 0; r < quarter_tiles; ++r) {
                    noc_async_read(in.get_noc_addr(base + r), ap, in_tile_bytes);
                    ap += in_tile_bytes;
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_a_landed, slot_tiles);

        for (uint32_t nb = 1; nb < num_n_blocks; ++nb) {
            load_weight_block(h, nb);
        }
    }
}
