// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "broadcast.hpp"
#include "rmsnorm_dm.hpp"

// Feeds the compute kernel's NormWork: the h rows of the next M block are read one tile row per K block step of the
// gate/up matmul (statistics) and of the down matmul (normalisation), mirroring the compute kernel's hooks.
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
    constexpr uint32_t M_blocks = CT_ARG("M_blocks");
    Broadcast bcast = BROADCAST();
    const uint32_t group_row = RT_ARG("group_row");
    const uint32_t own_blocks = RT_ARG("own_blocks");
    const uint32_t units = own_blocks * rows;
    const uint32_t stats_send_step = units + 3;

    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes);
    const auto w_norm = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_norm")>(), RT_ARG("w_norm_addr"), tile_bytes);

    CircularBuffer ring(gate_up.cb_in0);
    CircularBuffer norm_out(CT_ARG("cb_norm_out"));
    CircularBuffer a(gate_up.cb_out);
    CircularBuffer stats_out(CT_ARG("cb_stats_out"));
    CircularBuffer stats_in(CT_ARG("cb_stats_in"));
    const uint32_t stats_in_base = stats_in.get_write_ptr();

    auto read_unit = [&](uint32_t m_block, uint32_t unit) {
        const uint32_t block = unit / rows;
        read_h_row(
            noc,
            h,
            CT_ARG("cb_norm_in"),
            RT_ARG("M_start") + m_block * rows + unit % rows,
            gate_up.M_logical,
            gate_up.in0_row_tiles,
            (group_row + group * block) * K_block,
            K_block,
            tile_bytes);
    };
    auto push_weights = [&](uint32_t unit) {
        if (unit % rows == 0) {
            push_norm_weights(
                CT_ARG("cb_norm_weight_cache"),
                CT_ARG("cb_norm_weight"),
                (group_row + group * (unit / rows)) * K_block,
                K_block,
                tile_bytes);
        }
    };
    auto send_stats = [&](uint32_t m_block) {
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
    };
    auto receive_stats = [&](uint32_t m_block) {
        for (uint32_t source = 0; source < group; source++) {
            Semaphore<>(CT_ARG("sem_stats_arrival") + source).wait_min(m_block + 1);
        }
        stats_in.reserve_back(group * rows);
        stats_in.push_back(group * rows);
    };

    fill_reduce_scaler(noc, CT_ARG("cb_norm_scaler"), tile_bytes);
    load_norm_weights(noc, w_norm, CT_ARG("cb_norm_weight_cache"), gate_up.in0_row_tiles);

    for (uint32_t unit = 0; unit < units; unit++) {
        read_unit(0, unit);
    }
    send_stats(0);
    const uint32_t prefetched_units = units < CT_ARG("norm_in_units") ? units : CT_ARG("norm_in_units");
    for (uint32_t unit = 0; unit < units; unit++) {
        if (unit == prefetched_units) {
            receive_stats(0);
        }
        push_weights(unit);
        read_unit(0, unit);
    }

    for (uint32_t m_block = 0; m_block < M_blocks; m_block++) {
        const bool has_next = m_block + 1 < M_blocks;

        const uint32_t norm_base = norm_out.get_read_ptr();
        uint32_t step = 0;
        for (uint32_t n_block = 0; n_block < gate_up.N_blocks; n_block++) {
            for (uint32_t k_iter = 0; k_iter < gate_up.K_blocks; k_iter++, step++) {
                if (has_next) {
                    if (step < units) {
                        read_unit(m_block + 1, step);
                    }
                    if (step == stats_send_step) {
                        send_stats(m_block + 1);
                    }
                }
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
            if (has_next) {
                if (k == 0) {
                    receive_stats(m_block + 1);
                }
                if (k < units) {
                    push_weights(k);
                    read_unit(m_block + 1, k);
                }
            }
            bcast.step(noc, ring, block_tiles, block_bytes, k / a_blocks, [&] {
                const uint32_t slot = k % a_blocks;
                a.wait_front((slot + 1) * block_tiles);
                return a_base + slot * block_bytes;
            });
        }
        a.pop_front(a_blocks * block_tiles);
    }
}
