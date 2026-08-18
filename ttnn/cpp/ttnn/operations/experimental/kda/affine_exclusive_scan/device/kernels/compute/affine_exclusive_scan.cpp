// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace {
template <bool Mirror>
FORCE_INLINE void matmul(
    DataflowBuffer& a,
    DataflowBuffer& b,
    DataflowBuffer& out,
    DataflowBuffer& send,
    uint32_t Mt,
    uint32_t Kt,
    uint32_t Nt) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(Mt * Nt);
    if constexpr (Mirror) {
        send.reserve_back(Mt * Nt);
    }
    pack_reconfig_data_format(out_id);
    reconfig_data_format(b_id, a_id);
    matmul_init(a_id, b_id);
    for (uint32_t m = 0; m < Mt; m++) {
        for (uint32_t n = 0; n < Nt; n++) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; k++) {
                matmul_tiles(a_id, b_id, m * Kt + k, k * Nt + n, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_id, m * Nt + n);
            tile_regs_release();
        }
    }
    out.push_back(Mt * Nt);
    if constexpr (Mirror) {
        send.push_back(Mt * Nt);
    }
}

template <bool Mirror>
FORCE_INLINE void add(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& out, DataflowBuffer& send, uint32_t tiles) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(tiles);
    if constexpr (Mirror) {
        send.reserve_back(tiles);
    }
    pack_reconfig_data_format(out_id);
    reconfig_data_format(a_id, b_id);
    add_init(a_id, b_id);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        add_tiles(a_id, b_id, tile, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_id, tile);
        tile_regs_release();
    }
    out.push_back(tiles);
    if constexpr (Mirror) {
        send.push_back(tiles);
    }
}

template <bool Mirror>
FORCE_INLINE void copy(DataflowBuffer& in, DataflowBuffer& out, DataflowBuffer& send, uint32_t tiles) {
    const uint32_t in_id = in.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(tiles);
    if constexpr (Mirror) {
        send.reserve_back(tiles);
    }
    pack_reconfig_data_format(out_id);
    reconfig_data_format_srca(in_id);
    copy_tile_to_dst_init_short(in_id);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        copy_tile(in_id, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_id, tile);
        tile_regs_release();
    }
    out.push_back(tiles);
    if constexpr (Mirror) {
        send.push_back(tiles);
    }
}
}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    constexpr uint32_t kk = Kt * Kt;
    constexpr uint32_t kv = Kt * Vt;
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer stage_a_ping(dfb::stage_a_ping);
    DataflowBuffer stage_b_ping(dfb::stage_b_ping);
    DataflowBuffer stage_a_pong(dfb::stage_a_pong);
    DataflowBuffer stage_b_pong(dfb::stage_b_pong);
    DataflowBuffer send_a_ping(dfb::send_a_ping);
    DataflowBuffer send_b_ping(dfb::send_b_ping);
    DataflowBuffer send_a_pong(dfb::send_a_pong);
    DataflowBuffer send_b_pong(dfb::send_b_pong);
    DataflowBuffer remote_a(dfb::remote_a);
    DataflowBuffer remote_b(dfb::remote_b);
    DataflowBuffer initial_state(dfb::initial_state);
    DataflowBuffer final(dfb::final);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer stage_token(dfb::stage_token);

    compute_kernel_hw_startup(dfb::initial_a, dfb::initial_b, dfb::stage_a_ping);
    initial_a.wait_front(kk);
    initial_b.wait_front(kv);
    copy<true>(initial_a, stage_a_ping, send_a_ping, kk);
    copy<true>(initial_b, stage_b_ping, send_b_ping, kv);
    initial_a.pop_front(kk);
    initial_b.pop_front(kv);

    bool ping = false;
    for (uint32_t distance = 1; distance < G; distance *= 2) {
        if (group < distance) {
            continue;
        }
        DataflowBuffer& current_a = ping ? stage_a_pong : stage_a_ping;
        DataflowBuffer& current_b = ping ? stage_b_pong : stage_b_ping;
        DataflowBuffer& next_a = ping ? stage_a_ping : stage_a_pong;
        DataflowBuffer& next_b = ping ? stage_b_ping : stage_b_pong;
        DataflowBuffer& next_send_a = ping ? send_a_ping : send_a_pong;
        DataflowBuffer& next_send_b = ping ? send_b_ping : send_b_pong;
        stage_token.wait_front(1);
        current_a.wait_front(kk);
        current_b.wait_front(kv);
        remote_a.wait_front(kk);
        remote_b.wait_front(kv);
        matmul<true>(current_a, remote_a, next_a, next_send_a, Kt, Kt, Kt);
        matmul<false>(current_a, remote_b, scratch, scratch, Kt, Kt, Vt);
        scratch.wait_front(kv);
        add<true>(scratch, current_b, next_b, next_send_b, kv);
        stage_token.pop_front(1);
        current_a.pop_front(kk);
        current_b.pop_front(kv);
        remote_a.pop_front(kk);
        remote_b.pop_front(kv);
        scratch.pop_front(kv);
        ping = !ping;
    }

    stage_token.wait_front(1);
    initial_state.wait_front(kv);
    if (group == 0) {
        copy<false>(initial_state, final, final, kv);
    } else {
        remote_a.wait_front(kk);
        remote_b.wait_front(kv);
        matmul<false>(remote_a, initial_state, scratch, scratch, Kt, Kt, Vt);
        scratch.wait_front(kv);
        add<false>(scratch, remote_b, final, final, kv);
        remote_a.pop_front(kk);
        remote_b.pop_front(kv);
        scratch.pop_front(kv);
    }
    initial_state.pop_front(kv);
    stage_token.pop_front(1);
}
