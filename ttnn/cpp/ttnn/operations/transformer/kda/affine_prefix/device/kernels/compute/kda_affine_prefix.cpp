// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_initial_a = 0;
constexpr uint32_t cb_initial_b = 1;
constexpr uint32_t cb_stage_a_ping = 2;
constexpr uint32_t cb_stage_b_ping = 3;
constexpr uint32_t cb_stage_a_pong = 4;
constexpr uint32_t cb_stage_b_pong = 5;
constexpr uint32_t cb_remote_a = 6;
constexpr uint32_t cb_remote_b = 7;
constexpr uint32_t cb_initial_state = 8;
constexpr uint32_t cb_final = 9;
constexpr uint32_t cb_scratch = 10;
constexpr uint32_t cb_stage_token = 11;

inline void wait(uint32_t cb, uint32_t tiles) { CircularBuffer(cb).wait_front(tiles); }
inline void pop(uint32_t cb, uint32_t tiles) { CircularBuffer(cb).pop_front(tiles); }

void matmul(uint32_t a, uint32_t b, uint32_t out, uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    cb_reserve_back(out, Mt * Nt);
    pack_reconfig_data_format(out);
    reconfig_data_format(b, a);
    matmul_init(a, b);
    for (uint32_t m = 0; m < Mt; m++) {
        for (uint32_t n = 0; n < Nt; n++) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; k++) {
                matmul_tiles(a, b, m * Kt + k, k * Nt + n, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out, m * Nt + n);
            tile_regs_release();
        }
    }
    cb_push_back(out, Mt * Nt);
}

void add(uint32_t a, uint32_t b, uint32_t out, uint32_t tiles) {
    cb_reserve_back(out, tiles);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    add_tiles_init(a, b);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        add_tiles(a, b, tile, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, tiles);
}

void copy(uint32_t in, uint32_t out, uint32_t tiles) {
    cb_reserve_back(out, tiles);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        copy_tile(in, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, tile);
        tile_regs_release();
    }
    cb_push_back(out, tiles);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr bool compose_only = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t kk = Kt * Kt;
    constexpr uint32_t kv = Kt * Vt;
    const uint32_t group = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_initial_a, cb_initial_b, cb_final);
    wait(cb_initial_a, kk);
    wait(cb_initial_b, kv);
    copy(cb_initial_a, cb_stage_a_ping, kk);
    copy(cb_initial_b, cb_stage_b_ping, kv);
    pop(cb_initial_a, kk);
    pop(cb_initial_b, kv);

    bool ping = false;
    for (uint32_t distance = 1; distance < G; distance *= 2) {
        if (group < distance) {
            continue;
        }
        const uint32_t current_a = ping ? cb_stage_a_pong : cb_stage_a_ping;
        const uint32_t current_b = ping ? cb_stage_b_pong : cb_stage_b_ping;
        const uint32_t next_a = ping ? cb_stage_a_ping : cb_stage_a_pong;
        const uint32_t next_b = ping ? cb_stage_b_ping : cb_stage_b_pong;
        wait(cb_stage_token, 1);
        wait(current_a, kk);
        wait(current_b, kv);
        wait(cb_remote_a, kk);
        wait(cb_remote_b, kv);
        matmul(current_a, cb_remote_a, next_a, Kt, Kt, Kt);
        matmul(current_a, cb_remote_b, cb_scratch, Kt, Kt, Vt);
        wait(cb_scratch, kv);
        add(cb_scratch, current_b, next_b, kv);
        pop(cb_stage_token, 1);
        pop(current_a, kk);
        pop(current_b, kv);
        pop(cb_remote_a, kk);
        pop(cb_remote_b, kv);
        pop(cb_scratch, kv);
        ping = !ping;
    }

    if constexpr (compose_only) {
        return;
    }

    wait(cb_stage_token, 1);
    if (group == 0) {
        wait(cb_initial_state, kv);
        copy(cb_initial_state, cb_final, kv);
        pop(cb_initial_state, kv);
    } else {
        wait(cb_remote_a, kk);
        wait(cb_remote_b, kv);
        wait(cb_initial_state, kv);
        matmul(cb_remote_a, cb_initial_state, cb_scratch, Kt, Kt, Vt);
        wait(cb_scratch, kv);
        add(cb_scratch, cb_remote_b, cb_final, kv);
        pop(cb_remote_a, kk);
        pop(cb_remote_b, kv);
        pop(cb_initial_state, kv);
        pop(cb_scratch, kv);
    }
    pop(cb_stage_token, 1);
}
