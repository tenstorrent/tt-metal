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
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/identity_complement.hpp"

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

// Pack slot `staged` to the stage and send buffers and, with `out`, slot `published` to the output row.
template <typename DFBStage, typename DFBSend, typename DFBPub>
FORCE_INLINE void pack_tile_row_entry(
    DFBStage& stage, DFBSend& send, DFBPub* out, uint32_t tile, uint32_t staged, uint32_t published, uint32_t column) {
    if (out != nullptr) {
        pack_and_publish(stage, send, *out, tile, staged, published, column);
    } else {
        pack_tile(staged, stage.get_id(), tile);
        pack_tile(staged, send.get_id(), tile);
    }
}

// stage = send = a + b over Rows x Columns tiles. With `out`, the publishing group's last update of B also publishes
// each tile row.
template <
    uint32_t Rows,
    uint32_t Columns,
    typename DFBA,
    typename DFBB,
    typename DFBStage,
    typename DFBSend,
    typename DFBPub>
void add(DFBA& a, DFBB& b, DFBStage& stage, DFBSend& send, DFBPub* out) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    stage.reserve_back(Rows * Columns);
    send.reserve_back(Rows * Columns);
    // The preceding LLK operation is matmul. Establish add's source state before consuming its independently queued
    // operands; the packer remains configured for the canonical FP32 internal format.
    reconfig_data_format(a_id, b_id);
    add_init(a_id, b_id);
    for (uint32_t row = 0; row < Rows; row++) {
        if (out != nullptr) {
            out->reserve_back(Columns);
        }
        for (uint32_t column = 0; column < Columns; column++) {
            const uint32_t tile = row * Columns + column;
            tile_regs_acquire();
            add_tiles(a_id, b_id, tile, tile, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile_row_entry(stage, send, out, tile, 0, 0, column);
            tile_regs_release();
        }
        if (out != nullptr) {
            out->push_back(Columns);
        }
    }
    stage.push_back(Rows * Columns);
    send.push_back(Rows * Columns);
}

// Copy a group's Rows x Columns summary tiles to the stage and send buffers. With Complement the input is E (summaries
// arrive in complement form [E | B], E = A - I) and A = E + I is staged, formed on the diagonal tiles in FP32 DST from
// the identity tile (UnpackToDest, exact). With `out`, a single active group also publishes its summary unchanged, each
// tile row as it is staged: E stays in complement form for the output. Initial summaries may be BF16 while stage and
// send buffers use the canonical FP32 internal format; startup already configured the packer for the internal format.
template <
    uint32_t Rows,
    uint32_t Columns,
    bool Complement,
    typename DFBIn,
    typename DFBStage,
    typename DFBSend,
    typename DFBIdentity,
    typename DFBPub>
void copy_rows(DFBIn& in, DFBStage& stage, DFBSend& send, DFBIdentity& identity, DFBPub* out) {
    const uint32_t in_id = in.get_id();
    stage.reserve_back(Rows * Columns);
    send.reserve_back(Rows * Columns);
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
    for (uint32_t row = 0; row < Rows; row++) {
        if (out != nullptr) {
            out->reserve_back(Columns);
        }
        for (uint32_t column = 0; column < Columns; column++) {
            const uint32_t tile = row * Columns + column;
            tile_regs_acquire();
            copy_tile(in_id, tile, 0);
            uint32_t staged = 0;
            if (Complement && row == column) {
                kda::apply_identity<false>(identity, in_id, 0, 1, 2);
                copy_init(in_id);
                staged = 2;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile_row_entry(stage, send, out, tile, staged, 0, column);
            tile_regs_release();
        }
        if (out != nullptr) {
            out->push_back(Columns);
        }
    }
    stage.push_back(Rows * Columns);
    send.push_back(Rows * Columns);
}

// The last composition of A on the publishing group, publishing each tile row as E. A tile loop rather than
// matmul_product's matmul_block: E = A - I on a diagonal tile needs DST slots 1 and 2, which a block's output occupies.
// A stays in slot 0 for the stage and send buffers, and E lands in slot 2, so the subtraction is exact in FP32 DST
// before the single BF16 rounding of the pack.
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
            uint32_t published = 0;
            if (row == column) {
                // matmul's in1 (b) is SrcA.
                kda::apply_identity<true>(identity, b_id, 0, 1, 2);
                published = 2;
            }
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
    identity.wait_front(1);
    initial_a.wait_front(a_tiles);
    initial_b.wait_front(b_tiles);
    DataflowBuffer* const publish_initial = kda_chronology::publishes_unchanged(group, active) ? &output : nullptr;
    copy_rows<Kt, Kt, true>(initial_a, stage_a, send_a, identity, publish_initial);
    copy_rows<Kt, Vt, false>(initial_b, stage_b, send_b, identity, publish_initial);
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
        const bool publish_now = kda_chronology::publishes_at_stage(group, active, distance);
        // Both FP32 products remain separate calls: they consume different right-hand operands and publish
        // different output rectangles.
        if (publish_now) {
            matmul_product_published<Kt>(stage_a, remote_a, stage_a, send_a, identity, output);
        } else {
            matmul_product<Kt, Kt, Kt>(stage_a, remote_a, stage_a, &send_a);
        }
        matmul_product<Kt, Kt, Vt>(stage_a, remote_b, scratch, static_cast<decltype(&send_a)>(nullptr));
        scratch.wait_front(b_tiles);
        add<Kt, Vt>(scratch, stage_b, stage_b, send_b, publish_now ? &output : nullptr);
        stage_a.pop_front(a_tiles);
        stage_b.pop_front(b_tiles);
        remote_a.pop_front(a_tiles);
        remote_b.pop_front(b_tiles);
        scratch.pop_front(b_tiles);
    }
}
