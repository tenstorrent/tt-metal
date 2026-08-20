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

void matmul(
    DataflowBuffer& a,
    DataflowBuffer& b,
    DataflowBuffer& out,
    DataflowBuffer* send,
    uint32_t Mt,
    uint32_t Kt,
    uint32_t Nt) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t send_id = send == nullptr ? 0 : send->get_id();
    out.reserve_back(Mt * Nt);
    if (send != nullptr) {
        send->reserve_back(Mt * Nt);
    }
    for (uint32_t m = 0; m < Mt; m++) {
        for (uint32_t n = 0; n < Nt; n++) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; k++) {
                matmul_tiles(a_id, b_id, m * Kt + k, k * Nt + n, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_id, m * Nt + n);
            if (send != nullptr) {
                pack_tile(0, send_id, m * Nt + n);
            }
            tile_regs_release();
        }
    }
    out.push_back(Mt * Nt);
    if (send != nullptr) {
        send->push_back(Mt * Nt);
    }
}

void add(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& out, DataflowBuffer& send, uint32_t tiles) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t send_id = send.get_id();
    out.reserve_back(tiles);
    send.reserve_back(tiles);
    pack_reconfig_data_format(out_id);
    reconfig_data_format(a_id, b_id);
    add_init(a_id, b_id);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        add_tiles(a_id, b_id, tile, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_id, tile);
        pack_tile(0, send_id, tile);
        tile_regs_release();
    }
    out.push_back(tiles);
    send.push_back(tiles);
}

void copy(DataflowBuffer& in, DataflowBuffer& out, DataflowBuffer& send, uint32_t tiles) {
    const uint32_t in_id = in.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t send_id = send.get_id();
    out.reserve_back(tiles);
    send.reserve_back(tiles);
    pack_reconfig_data_format(out_id);
    reconfig_data_format_srca(in_id);
    copy_tile_to_dst_init_short(in_id);
    for (uint32_t tile = 0; tile < tiles; tile++) {
        tile_regs_acquire();
        copy_tile(in_id, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_id, tile);
        pack_tile(0, send_id, tile);
        tile_regs_release();
    }
    out.push_back(tiles);
    send.push_back(tiles);
}

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    constexpr uint32_t kk = Kt * Kt;
    constexpr uint32_t kv = Kt * Vt;
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer stage_a(dfb::stage_a);
    DataflowBuffer stage_b(dfb::stage_b);
    DataflowBuffer send_a(dfb::send_a);
    DataflowBuffer send_b(dfb::send_b);
    DataflowBuffer remote_a(dfb::remote_a);
    DataflowBuffer remote_b(dfb::remote_b);
    DataflowBuffer scratch(dfb::scratch);

    compute_kernel_hw_startup(dfb::initial_a, dfb::initial_b, dfb::stage_a);
    initial_a.wait_front(kk);
    initial_b.wait_front(kv);
    copy(initial_a, stage_a, send_a, kk);
    copy(initial_b, stage_b, send_b, kv);
    initial_a.pop_front(kk);
    initial_b.pop_front(kv);

    for (uint32_t distance = 1; distance < G; distance *= 2) {
        if (group < distance) {
            continue;
        }
        stage_a.wait_front(kk);
        stage_b.wait_front(kv);
        remote_a.wait_front(kk);
        remote_b.wait_front(kv);
        pack_reconfig_data_format(stage_a.get_id());
        reconfig_data_format(remote_a.get_id(), stage_a.get_id());
        matmul_init(stage_a.get_id(), remote_a.get_id());
        matmul(stage_a, remote_a, stage_a, &send_a, Kt, Kt, Kt);
        matmul(stage_a, remote_b, scratch, nullptr, Kt, Kt, Vt);
        scratch.wait_front(kv);
        add(scratch, stage_b, stage_b, send_b, kv);
        stage_a.pop_front(kk);
        stage_b.pop_front(kv);
        remote_a.pop_front(kk);
        remote_b.pop_front(kv);
        scratch.pop_front(kv);
    }
}
