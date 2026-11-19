// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "broadcast.hpp"
#include "rmsnorm_dm.hpp"

void kernel_main() {
    Noc noc;
    constexpr MatmulShape gate_up = MATMUL_SHAPE("mm1");
    constexpr MatmulShape down = MATMUL_SHAPE("mm2");
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t rows = gate_up.M_block;
    constexpr uint32_t K_block = gate_up.K_block;
    constexpr uint32_t block_tiles = rows * K_block;
    constexpr uint32_t block_bytes = block_tiles * tile_bytes;
    constexpr uint32_t group = CT_ARG("group_size");
    constexpr uint32_t a_blocks = gate_up.N_blocks;
    Broadcast bcast = BROADCAST();
    const uint32_t group_row = RT_ARG("group_row");
    const uint32_t own_blocks = RT_ARG("own_blocks");

    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes);
    const auto w_norm = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_norm")>(), RT_ARG("w_norm_addr"), tile_bytes);

    CircularBuffer ring(gate_up.cb_in0);
    CircularBuffer norm_out(CT_ARG("cb_norm_out"));
    CircularBuffer a(gate_up.cb_out);
    CircularBuffer stats_out(CT_ARG("cb_stats_out"));
    CircularBuffer stats_in(CT_ARG("cb_stats_in"));
    const uint32_t stats_in_base = stats_in.get_write_ptr();

    fill_reduce_scaler(noc, CT_ARG("cb_norm_scaler"), tile_bytes);
    load_norm_weights(noc, w_norm, CT_ARG("cb_norm_weight_cache"), gate_up.in0_row_tiles);

    for (uint32_t m_block = 0; m_block < CT_ARG("M_blocks"); m_block++) {
        const uint32_t m_tile = RT_ARG("M_start") + m_block * rows;
        for (uint32_t b = 0; b < own_blocks; b++) {
            read_h_block_rows(
                noc,
                h,
                CT_ARG("cb_norm_in"),
                m_tile,
                rows,
                gate_up.M_logical,
                gate_up.in0_row_tiles,
                (group_row + group * b) * K_block,
                K_block,
                tile_bytes);
        }

        stats_out.wait_front(rows);
        Semaphore<> own_arrival(CT_ARG("sem_stats_arrival") + group_row);
        bcast.multicast(
            noc,
            stats_out.get_read_ptr(),
            stats_in_base + group_row * rows * tile_bytes,
            rows * tile_bytes,
            own_arrival,
            m_block + 1);
        stats_out.pop_front(rows);
        for (uint32_t source = 0; source < group; source++) {
            Semaphore<>(CT_ARG("sem_stats_arrival") + source).wait_min(m_block + 1);
        }
        stats_in.reserve_back(group * rows);
        stats_in.push_back(group * rows);

        for (uint32_t b = 0; b < own_blocks; b++) {
            const uint32_t k_tile = (group_row + group * b) * K_block;
            push_norm_weights(CT_ARG("cb_norm_weight_cache"), CT_ARG("cb_norm_weight"), k_tile, K_block, tile_bytes);
            read_h_block_rows(
                noc,
                h,
                CT_ARG("cb_norm_in"),
                m_tile,
                rows,
                gate_up.M_logical,
                gate_up.in0_row_tiles,
                k_tile,
                K_block,
                tile_bytes);
        }

        const uint32_t norm_base = norm_out.get_read_ptr();
        for (uint32_t n_block = 0; n_block < gate_up.N_blocks; n_block++) {
            for (uint32_t k_iter = 0; k_iter < gate_up.K_blocks; k_iter++) {
                if (n_block > 0 && k_iter == 0) {
                    continue;
                }
                const uint32_t k = n_block % 2 == 0 ? k_iter : gate_up.K_blocks - 1 - k_iter;
                bcast.step(noc, ring, block_tiles, block_bytes, k % group, [&] {
                    const uint32_t slot = k / group;
                    norm_out.wait_front((slot + 1) * block_tiles);
                    return norm_base + slot * block_bytes;
                });
            }
        }
        norm_out.pop_front(CT_ARG("max_own_blocks") * block_tiles);

        const uint32_t a_base = a.get_read_ptr();
        for (uint32_t k = 0; k < down.K_blocks; k++) {
            bcast.step(noc, ring, block_tiles, block_bytes, k / a_blocks, [&] {
                const uint32_t slot = k % a_blocks;
                a.wait_front((slot + 1) * block_tiles);
                return a_base + slot * block_bytes;
            });
        }
        a.pop_front(a_blocks * block_tiles);
    }
}
