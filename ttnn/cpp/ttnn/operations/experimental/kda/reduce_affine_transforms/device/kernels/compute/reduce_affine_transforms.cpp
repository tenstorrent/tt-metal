// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// FP32 half-DST holds four 32x32 output tiles. Production's four-tile output rows fit exactly and use
// matmul_block; wider rows retain the tile loop because a row-major B operand cannot be column-sliced as a block
// without repacking.
template <uint32_t Mt, uint32_t Kt, uint32_t Nt, typename DFBA, typename DFBB, typename DFBOut, typename DFBSend>
void matmul_product(DFBA& a, DFBB& b, DFBOut& out, DFBSend* send) {
    constexpr uint32_t max_block_columns = 4;
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t send_id = send == nullptr ? 0 : send->get_id();
    out.reserve_back(Mt * Nt);
    if (send != nullptr) {
        send->reserve_back(Mt * Nt);
    }
    if constexpr (Nt <= max_block_columns) {
        matmul_block_init(a_id, b_id, false, Nt, 1, Kt);
        for (uint32_t row = 0; row < Mt; row++) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; k++) {
                matmul_block(a_id, b_id, row * Kt + k, k * Nt, 0, false, Nt, 1, Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t column = 0; column < Nt; column++) {
                const uint32_t out_tile = row * Nt + column;
                pack_tile(column, out_id, out_tile);
                if (send != nullptr) {
                    pack_tile(column, send_id, out_tile);
                }
            }
            tile_regs_release();
        }
    } else {
        matmul_init(a_id, b_id);
        for (uint32_t row = 0; row < Mt; row++) {
            for (uint32_t column = 0; column < Nt; column++) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < Kt; k++) {
                    matmul_tiles(a_id, b_id, row * Kt + k, k * Nt + column, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                const uint32_t out_tile = row * Nt + column;
                pack_tile(0, out_id, out_tile);
                if (send != nullptr) {
                    pack_tile(0, send_id, out_tile);
                }
                tile_regs_release();
            }
        }
    }
    out.push_back(Mt * Nt);
    if (send != nullptr) {
        send->push_back(Mt * Nt);
    }
}

template <typename DFBA, typename DFBB, typename DFBOut, typename DFBSend>
void add(DFBA& a, DFBB& b, DFBOut& out, DFBSend& send, uint32_t tiles) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t send_id = send.get_id();
    out.reserve_back(tiles);
    send.reserve_back(tiles);
    // The preceding LLK operation is matmul. Establish add's source state before consuming its independently queued
    // operands; the packer remains configured for the canonical FP32 internal format.
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

template <typename DFBIn, typename DFBOut, typename DFBSend>
void copy(DFBIn& in, DFBOut& out, DFBSend& send, uint32_t tiles) {
    const uint32_t in_id = in.get_id();
    const uint32_t out_id = out.get_id();
    const uint32_t send_id = send.get_id();
    out.reserve_back(tiles);
    send.reserve_back(tiles);
    // Initial summaries may be BF16 while stage and send buffers use the canonical FP32 internal format. Copy updates
    // the source format; startup already configured the packer for the internal format.
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
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

// Summaries arrive in complement form [E | B] (E = A - I). Load one E tile into DST slot 0 and, on a diagonal tile,
// form A = E + I in FP32 DST in slot 2 from the identity tile in slot 1 (UnpackToDest, exact). Returns the slot holding
// A; the caller re-initializes its own unpack and math state afterwards.
template <typename DFBIdentity>
FORCE_INLINE uint32_t load_complement_tile(uint32_t in_id, uint32_t tile, DFBIdentity& identity, bool diagonal) {
    copy_init(in_id);
    copy_tile(in_id, tile, 0);
    if (!diagonal) {
        return 0;
    }
    const uint32_t identity_id = identity.get_id();
    reconfig_data_format_srca(in_id, identity_id);
    copy_init(identity_id);
    copy_tile(identity_id, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 2);
    reconfig_data_format_srca(identity_id, in_id);
    return 2;
}

// Copy a group's E rows to the stage and send buffers as A = E + I in FP32.
template <uint32_t Kt, typename DFBIn, typename DFBOut, typename DFBSend, typename DFBIdentity>
void copy_complement(DFBIn& in, DFBOut& out, DFBSend& send, DFBIdentity& identity) {
    const uint32_t in_id = in.get_id();
    out.reserve_back(Kt * Kt);
    send.reserve_back(Kt * Kt);
    reconfig_data_format_srca(in_id);
    for (uint32_t row = 0; row < Kt; row++) {
        for (uint32_t column = 0; column < Kt; column++) {
            const uint32_t tile = row * Kt + column;
            tile_regs_acquire();
            const uint32_t a_slot = load_complement_tile(in_id, tile, identity, row == column);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(a_slot, out.get_id(), tile);
            pack_tile(a_slot, send.get_id(), tile);
            tile_regs_release();
        }
    }
    out.push_back(Kt * Kt);
    send.push_back(Kt * Kt);
}

// The last group publishes its final transform once, as the packed BF16 rows [E | B] the distributed prefix gathers,
// beside the FP32 A and B that its stage and send buffers keep. E = A - I is formed in FP32 DST: A stays in slot 0 for
// the stage and send buffers, the identity tile (UnpackToDest, exact) goes to slot 1, and on diagonal tiles E lands in
// slot 2, so the subtraction is exact before the single BF16 rounding of the pack. Returns the slot to publish; the
// caller re-initializes its own unpack and math state afterwards.
template <typename DFBIdentity>
FORCE_INLINE uint32_t complement_on_diagonal(DFBIdentity& identity, uint32_t srca_id, bool diagonal) {
    if (!diagonal) {
        return 0;
    }
    const uint32_t identity_id = identity.get_id();
    reconfig_data_format_srca(srca_id, identity_id);
    copy_init(identity_id);
    copy_tile(identity_id, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 2);
    reconfig_data_format_srca(identity_id, srca_id);
    return 2;
}

// Pack slot `staged` (FP32) to the stage and send buffers and slot `published` to the BF16 output row.
template <typename DFBStage, typename DFBSend, typename DFBOut>
FORCE_INLINE void pack_and_publish(
    DFBStage& stage, DFBSend& send, DFBOut& out, uint32_t tile, uint32_t staged, uint32_t published, uint32_t column) {
    pack_tile(staged, stage.get_id(), tile);
    pack_tile(staged, send.get_id(), tile);
    pack_reconfig_data_format(out.get_id());
    pack_tile(published, out.get_id(), column);
    pack_reconfig_data_format(stage.get_id());
}

// The last composition of A on the publishing group: matmul_product's tile loop, publishing each tile row as E.
template <
    uint32_t Kt,
    typename DFBA,
    typename DFBB,
    typename DFBOut,
    typename DFBSend,
    typename DFBIdentity,
    typename DFBPub>
void matmul_product_published(DFBA& a, DFBB& b, DFBOut& stage, DFBSend& send, DFBIdentity& identity, DFBPub& out) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    stage.reserve_back(Kt * Kt);
    send.reserve_back(Kt * Kt);
    for (uint32_t row = 0; row < Kt; row++) {
        out.reserve_back(Kt);
        for (uint32_t column = 0; column < Kt; column++) {
            tile_regs_acquire();
            matmul_init(a_id, b_id);
            for (uint32_t k = 0; k < Kt; k++) {
                matmul_tiles(a_id, b_id, row * Kt + k, k * Kt + column, 0);
            }
            // matmul's in1 (b) is SrcA.
            const uint32_t published = complement_on_diagonal(identity, b_id, row == column);
            tile_regs_commit();
            tile_regs_wait();
            pack_and_publish(stage, send, out, row * Kt + column, 0, published, column);
            tile_regs_release();
        }
        out.push_back(Kt);
    }
    stage.push_back(Kt * Kt);
    send.push_back(Kt * Kt);
}

// The last update of B on the publishing group: add, publishing each tile row of B.
template <uint32_t Kt, uint32_t Vt, typename DFBA, typename DFBB, typename DFBOut, typename DFBSend, typename DFBPub>
void add_published(DFBA& a, DFBB& b, DFBOut& stage, DFBSend& send, DFBPub& out) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    stage.reserve_back(Kt * Vt);
    send.reserve_back(Kt * Vt);
    reconfig_data_format(a_id, b_id);
    add_init(a_id, b_id);
    for (uint32_t row = 0; row < Kt; row++) {
        out.reserve_back(Vt);
        for (uint32_t column = 0; column < Vt; column++) {
            const uint32_t tile = row * Vt + column;
            tile_regs_acquire();
            add_tiles(a_id, b_id, tile, tile, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_and_publish(stage, send, out, tile, 0, 0, column);
            tile_regs_release();
        }
        out.push_back(Vt);
    }
    stage.push_back(Kt * Vt);
    send.push_back(Kt * Vt);
}

// A single active group publishes its own summary unchanged, each tile row as it is staged: E stays in complement form
// for the output while A = E + I is staged; B is copied.
template <
    uint32_t Rows,
    uint32_t Columns,
    bool Complement,
    typename DFBIn,
    typename DFBOut,
    typename DFBSend,
    typename DFBIdentity,
    typename DFBPub>
void copy_published(DFBIn& in, DFBOut& stage, DFBSend& send, DFBIdentity& identity, DFBPub& out) {
    const uint32_t in_id = in.get_id();
    stage.reserve_back(Rows * Columns);
    send.reserve_back(Rows * Columns);
    reconfig_data_format_srca(in_id);
    for (uint32_t row = 0; row < Rows; row++) {
        out.reserve_back(Columns);
        for (uint32_t column = 0; column < Columns; column++) {
            const uint32_t tile = row * Columns + column;
            tile_regs_acquire();
            const uint32_t staged = load_complement_tile(in_id, tile, identity, Complement && row == column);
            tile_regs_commit();
            tile_regs_wait();
            pack_and_publish(stage, send, out, tile, staged, 0, column);
            tile_regs_release();
        }
        out.push_back(Columns);
    }
    stage.push_back(Rows * Columns);
    send.push_back(Rows * Columns);
}

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void compute(uint32_t group) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::initial_a, dfb::initial_b, dfb::stage_a);

    constexpr uint32_t a_tiles = Kt * Kt;
    constexpr uint32_t b_tiles = Kt * Vt;
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer stage_a(dfb::stage_a);
    DataflowBuffer stage_b(dfb::stage_b);
    DataflowBuffer send_a(dfb::send_a);
    DataflowBuffer send_b(dfb::send_b);
    DataflowBuffer remote_a(dfb::remote_a);
    DataflowBuffer remote_b(dfb::remote_b);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer identity(dfb::identity);
    DataflowBuffer output(dfb::output);

    kda_chronology::Topology topology{};
    {
        DataflowBuffer chronology(dfb::chronology_compute);
        topology = kda_chronology::receive(chronology);
    }
    const uint32_t active = topology.head_groups(G);
    if (group >= active) {
        return;
    }
    // The last active group composes the whole head and publishes it, in its last step.
    const bool publishes = group + 1 == active;
    identity.wait_front(1);
    initial_a.wait_front(a_tiles);
    initial_b.wait_front(b_tiles);
    if (publishes && active == 1) {
        copy_published<Kt, Kt, true>(initial_a, stage_a, send_a, identity, output);
        copy_published<Kt, Vt, false>(initial_b, stage_b, send_b, identity, output);
    } else {
        copy_complement<Kt>(initial_a, stage_a, send_a, identity);
        copy(initial_b, stage_b, send_b, b_tiles);
    }
    initial_a.pop_front(a_tiles);
    initial_b.pop_front(b_tiles);

    for (uint32_t distance = 1; distance < active; distance *= 2) {
        // Every participating group produces a prefix consumed by a later group at a subsequent power-of-two
        // distance. Only the final group writes DRAM, but these intermediate prefixes are required inputs.
        if (group < distance) {
            continue;
        }
        // Stage buffers are durable state shared across independently progressing PACK, UNPACK, and NoC stages. The
        // current destination-register lifetime cannot span that synchronization boundary, so each prefix is queued
        // and reacquired here.
        stage_a.wait_front(a_tiles);
        stage_b.wait_front(b_tiles);
        remote_a.wait_front(a_tiles);
        remote_b.wait_front(b_tiles);
        reconfig_data_format(remote_a.get_id(), stage_a.get_id());
        // The publishing group's last step is its last stage: every later distance exceeds its index.
        const bool publish_now = publishes && 2 * distance >= active;
        // Both FP32 products remain separate calls: they consume different right-hand operands and publish
        // different output rectangles.
        if (publish_now) {
            matmul_product_published<Kt>(stage_a, remote_a, stage_a, send_a, identity, output);
        } else {
            matmul_product<Kt, Kt, Kt>(stage_a, remote_a, stage_a, &send_a);
        }
        matmul_product<Kt, Kt, Vt>(stage_a, remote_b, scratch, static_cast<decltype(&send_a)>(nullptr));
        scratch.wait_front(b_tiles);
        if (publish_now) {
            add_published<Kt, Vt>(scratch, stage_b, stage_b, send_b, output);
        } else {
            add(scratch, stage_b, stage_b, send_b, b_tiles);
        }
        stage_a.pop_front(a_tiles);
        stage_b.pop_front(b_tiles);
        remote_a.pop_front(a_tiles);
        remote_b.pop_front(b_tiles);
        scratch.pop_front(b_tiles);
    }
}
