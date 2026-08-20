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

FORCE_INLINE void matmul(
    DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& out, uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(Mt * Nt);
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
}

FORCE_INLINE void add(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& out, uint32_t tiles) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(tiles);
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
}

FORCE_INLINE void copy(DataflowBuffer& in, DataflowBuffer& out, uint32_t tiles) {
    const uint32_t in_id = in.get_id();
    const uint32_t out_id = out.get_id();
    out.reserve_back(tiles);
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
}

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    constexpr uint32_t kk = Kt * Kt;
    constexpr uint32_t kv = Kt * Vt;
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer local_a(dfb::local_a);
    DataflowBuffer local_b(dfb::local_b);
    DataflowBuffer to_remote_a(dfb::to_remote_a);
    DataflowBuffer to_remote_b(dfb::to_remote_b);
    DataflowBuffer from_remote_a(dfb::from_remote_a);
    DataflowBuffer from_remote_b(dfb::from_remote_b);
    DataflowBuffer initial_state(dfb::initial_state);
    DataflowBuffer final(dfb::final);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer stage_token(dfb::stage_token);

    compute_kernel_hw_startup(dfb::initial_a, dfb::initial_b, dfb::to_remote_a);
    initial_a.wait_front(kk);
    initial_b.wait_front(kv);
    copy(initial_a, to_remote_a, kk);
    copy(initial_b, to_remote_b, kv);
    initial_a.pop_front(kk);
    initial_b.pop_front(kv);

    for (uint32_t distance = 1; distance < G; distance *= 2) {
        if (group < distance) {
            continue;
        }
        stage_token.wait_front(1);
        local_a.wait_front(kk);
        local_b.wait_front(kv);
        from_remote_a.wait_front(kk);
        from_remote_b.wait_front(kv);
        matmul(local_a, from_remote_a, to_remote_a, Kt, Kt, Kt);
        matmul(local_a, from_remote_b, scratch, Kt, Kt, Vt);
        scratch.wait_front(kv);
        add(scratch, local_b, to_remote_b, kv);
        stage_token.pop_front(1);
        local_a.pop_front(kk);
        local_b.pop_front(kv);
        from_remote_a.pop_front(kk);
        from_remote_b.pop_front(kv);
        scratch.pop_front(kv);
    }

    stage_token.wait_front(1);
    initial_state.wait_front(kv);
    if (group == 0) {
        copy(initial_state, final, kv);
    } else {
        from_remote_a.wait_front(kk);
        from_remote_b.wait_front(kv);
        matmul(from_remote_a, initial_state, scratch, Kt, Kt, Vt);
        scratch.wait_front(kv);
        add(scratch, from_remote_b, final, kv);
        from_remote_a.pop_front(kk);
        from_remote_b.pop_front(kv);
        scratch.pop_front(kv);
    }
    initial_state.pop_front(kv);
    stage_token.pop_front(1);
}
