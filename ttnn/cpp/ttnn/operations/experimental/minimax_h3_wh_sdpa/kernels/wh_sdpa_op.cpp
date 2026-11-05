// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The WH SDPA op's compute kernel (wh_sdpa_op_common.h), ported from tt-lab-private h3-wh src/trisc_wh_sdpa_op.cpp:
// the core loop of wh_sdpa_core.h on the three TRISCs, which walk the core's rounds; a round with a query block for
// this core is an item, the others only advance TRISC0 through the group's K/V stream.
//
// An item: Q lands from DRAM as [d slice][64 queries][16 d]; TRISC0 unpacks it with every face transposed (Haloize),
// TRISC1 moves it from SrcA into Dst half 1 and TRISC2 packs it to L1 as Q^T. Then the core loop runs the item's key
// blocks, block g of the stream from K/V buffer g % bufs once its reader has cast it there. At the end TRISC2 packs O^T
// / l, TRISC0 unpacks it transposed, TRISC1 moves it into Dst half 0 as two tiles of [8 d slices][32 queries][16 d] and
// TRISC2 packs that to the item's O staging, which the I/O kernel writes out.
//
// The kernel owns the Tensix while it runs; it saves the configuration space and every thread's GPRs at entry and puts
// them back, with SFPU LReg 11 (-1.0 by tt-metal convention), before it returns.

#include <cstdint>
#include "api/compute/common.h"
#include "wh_sdpa_core.h"
#include "wh_sdpa_op_common.h"

namespace {

constexpr uint32_t gpr_mark = 60u;

struct Op {
    uint32_t scratch, go, bufs, blocks, valid_last, partial_block, rounds, group_cores, group_index;
    uint32_t debug;
};

Op op;

volatile uint32_t& sync_word(uint32_t line) { return WH_REG(whs_line(op.scratch, line)); }

uint32_t round_word(uint32_t r) { return get_arg_val<uint32_t>(WHS_A_ROUND0 + r); }

bool idle_round(uint32_t r) { return (round_word(r) & 0xFFFFu) == WHS_IDLE || (op.debug & WHS_DEBUG_FEED_ONLY); }

inline void poll_pause() {
    for (uint32_t i = 0; i < 32u; ++i) {
        __asm__ volatile("nop");
    }
}

// Waits for a sync line to reach `value`, adding the cycles waited to stats line `stat`.
[[maybe_unused]] void wait_at_least(uint32_t line, uint32_t value, uint32_t stat) {
    if (sync_word(line) >= value) {
        return;
    }
    const uint32_t start = WH_REG(wh_wall_clock_lo);
    while (sync_word(line) < value) {
        poll_pause();
    }
    sync_word(WHS_S_STATS + stat) += WH_REG(wh_wall_clock_lo) - start;
}

// Waits until this thread's earlier instructions have passed the units in `wait`, through GPR gpr_mark.
[[maybe_unused]] void drain(uint32_t wait, uint32_t mark) {
    volatile uint32_t& word = wh_gpr(gpr_mark);
    wh_push(TT_OP_STALLWAIT(wh_block_thcon | wh_block_sync, wait));
    wh_set_gpr(gpr_mark, mark);
    while (word != mark) {
    }
}

uint32_t unpack0() { return op.scratch + WHS_UNPACK0; }

#if TT_TRISC_INDEX == 0

void setup() { setup_unpack<whp_sdpa_head128>(unpack0(), op.scratch + WHS_KV); }

bool block_ready(uint32_t g) { return sync_word(WHS_S_READY + g % op.bufs) >= g / op.bufs + 1u; }

void wait_block(uint32_t g) { wait_at_least(WHS_S_READY + g % op.bufs, g / op.bufs + 1u, 0u); }

// Block g for this core's fills. A wait ends on every waiting core of the group when the block's ready word lands, so
// the core then holds off `skew` cycles more: fills restarting in step on a group's cores wedge edge-row Tensix
// (tt-lab-private notes/minimax-h3-wh-compute.md, "Edge-row wedges").
void wait_block_skewed(uint32_t g, uint32_t skew) {
    if (block_ready(g)) {
        return;
    }
    wait_block(g);
    const uint32_t start = WH_REG(wh_wall_clock_lo);
    while (WH_REG(wh_wall_clock_lo) - start < skew) {
    }
}

void run() {
    const uint32_t blocks = op.blocks, bufs = op.bufs;
    const uint32_t skew = op.group_index * 2400u / op.group_cores;
    uint32_t g = 0, item = 0, mark = 0;
    for (uint32_t r = 0; r < op.rounds; ++r) {
        if (idle_round(r)) {
            for (uint32_t j = 0; j < blocks; ++j, ++g) {
                wait_block(g);
                sync_word(WHS_S_UNPACKED) = g + 1u;
            }
            continue;
        }
        wait_at_least(WHS_S_Q_LANDED, item + 1u, 1u);
        unpack_transposed(WHS_W_Q + item % 2u, 8u);
        wh_push(TT_OP_SEMWAIT(wh_block_sync | wh_block_unpack, sem_qt, wh_sem_on_zero));
        wh_push(TT_OP_SEMGET(sem_qt));
        drain(wh_wait_unpack0, ++mark);
        sync_word(WHS_S_Q_TAKEN) = item + 1u;
        wait_block_skewed(g, skew);
        unpack_qk<whp_sdpa_head128>(g % bufs);
        for (uint32_t j = 0; j < blocks; ++j, ++g) {
            if (j + 1u < blocks) {
                wait_block_skewed(g + 1u, skew);
                unpack_qk<whp_sdpa_head128>((g + 1u) % bufs);
            }
            unpack_pv<whp_sdpa_head128>(j % 2u, g % bufs);
            drain(wh_wait_unpack0 | wh_wait_unpack1, ++mark);
            sync_word(WHS_S_UNPACKED) = g + 1u;
        }
        wh_push(TT_OP_SEMWAIT(wh_block_sync | wh_block_unpack, sem_packed, wh_sem_on_zero));
        wh_push(TT_OP_SEMGET(sem_packed));
        unpack_transposed(WHS_W_OT, 8u);
        ++item;
    }
}

void cleanup() { wh_thread_reset(); }

#elif TT_TRISC_INDEX == 1

void setup() { setup_math(); }

// Q^T's chunk k (d slice k) to Dst rows 64 k on; O's chunk k (d slice k, 64 queries) to rows 256 t + 32 k on for
// query tile t, so that each tile packs as [8 d slices][32 queries][16 d].
constexpr uint32_t q_row(uint32_t k, uint32_t i) { return 64u * k + 8u * i; }

constexpr uint32_t o_row(uint32_t k, uint32_t i) { return 256u * (i / 4u) + 32u * k + 8u * (i % 4u); }

void run() {
    const uint32_t blocks = op.blocks, phases = 2u;
    bool first = true;
    for (uint32_t r = 0; r < op.rounds; ++r) {
        if (idle_round(r)) {
            continue;
        }
        move_chunks<q_row>(s_base(0u), 8u);
        qk<whp_sdpa_head128>(0u, true, first, phases);
        first = false;
        for (uint32_t j = 0; j < blocks; ++j) {
            if (j + 1u < blocks) {
                qk<whp_sdpa_head128>((j + 1u) % 2u, true, false, phases);
            }
            pv<whp_sdpa_head128>(phases, j == 0u);
        }
        // The last block's softmax post, which no QK takes.
        wh_push(TT_OP_SEMWAIT(wh_block_sync | wh_block_math, sem_softmax, wh_sem_on_zero));
        wh_push(TT_OP_SEMGET(sem_softmax));
        move_chunks<o_row>(o_base(0u), 8u);
    }
}

void cleanup() { wh_thread_reset(); }

#else

void setup() { setup_softmax(); }

void run() {
    const uint32_t blocks = op.blocks;
    const uint32_t p_base = unpack0() + WHS_W_P * WHS_W_BYTES;
    uint32_t item = 0, mark = 0;
    for (uint32_t r = 0; r < op.rounds; ++r) {
        if (idle_round(r)) {
            continue;
        }
        pack_moved(s_base(0u), unpack0() + WHS_W_QT * WHS_W_BYTES, 2u);
        push(TT_OP_SEMPOST(sem_qt));
        push(TT_OP_SEMPOST(sem_softmax));
        pack_format(true);
        reset_records();
        for (uint32_t j = 0; j < blocks; ++j) {
            const uint32_t valid = j == op.partial_block ? op.valid_last : WHS_ROWS;
            const uint32_t p = p_base + j % 2u * WH_SDPA_REGION_BYTES;
            if (j % 2u) {
                block<whp_sdpa_head128, 1u>(p, true, valid, false, true);
            } else {
                block<whp_sdpa_head128, 0u>(p, j != 0u, valid, false, true);
            }
        }
        finish<whp_sdpa_head128>(blocks);
        push(TT_OP_STALLWAIT(wh_block_pack | wh_block_sync | wh_block_cfg | wh_block_thcon, wh_wait_sfpu));
        pack_format(false);
#pragma GCC unroll 16
        for (uint32_t h = 0; h < 2u; ++h) {
            wh_pack_region(
                gpr_out, unpack0() + WHS_W_OT * WHS_W_BYTES + h * WH_SDPA_REGION_BYTES, WHS_FACE, o_base(h) / 16u);
        }
        push(TT_OP_STALLWAIT(wh_block_sync, wh_wait_pack));
        push(TT_OP_SEMPOST(sem_packed));
        if (item >= 2u) {
            wait_at_least(WHS_S_O_WRITTEN, item - 1u, 2u);
        }
        pack_moved(o_base(0u), op.scratch + WHS_O_STAGE + item % 2u * WHS_O_STAGE_BYTES, 2u);
        drain(wh_wait_pack, ++mark);
        sync_word(WHS_S_O_STAGED) = ++item;
    }
}

void cleanup() {
    push(TT_OP_STALLWAIT(wh_block_sync | wh_block_cfg | wh_block_thcon, wh_wait_pack));
    // LReg 11 is -1.0 by tt-metal's convention (set once at boot) and LLK code relies on it; the core loop's
    // broadcasts (park_faces) overwrite LRegs 11-14.
    push(TT_OP_SFPLOADI(0u, loadi_bf16, 0xBF80u));
    push(TT_OP_SFPCONFIG(0u, 11u, 0u));
    // sfpi code assumes the SFPU condition codes enabled with every lane's flag set between v_ifs (the boot state);
    // the softmax's lanes_masked(false) leaves them disabled, and the next kernel's first v_if then ran its body on
    // every lane (binary_ng's bf16 multiply zeroed rows 0-3 of its first face through `v_if(in0 == 0 || in1 == 0)`).
    push(TT_OP_SFPENCC(3u, 0u, 0u, 10u));
    wh_thread_reset();
}

#endif

}  // namespace

void kernel_main() {
    op.scratch = get_arg_val<uint32_t>(WHS_A_SCRATCH);
    op.go = get_arg_val<uint32_t>(WHS_A_GO);
    op.bufs = get_arg_val<uint32_t>(WHS_A_BUFS);
    op.rounds = get_arg_val<uint32_t>(WHS_A_ROUNDS);
    op.group_cores = get_arg_val<uint32_t>(WHS_A_GROUP_CORES);
    op.group_index = get_arg_val<uint32_t>(WHS_A_GROUP_INDEX);
    op.debug = get_arg_val<uint32_t>(WHS_A_DEBUG);

    // The I/O kernel zeroes the sync lines and meets the group, then posts this launch's token.
    while (sync_word(WHS_S_GO) != op.go) {
    }
    const uint32_t valid_keys = sync_word(WHS_S_VALID);
    op.blocks = whs_key_blocks(valid_keys);
    op.valid_last = valid_keys - (op.blocks - 1u) * WHS_ROWS;
    op.partial_block = op.valid_last < WHS_ROWS ? op.blocks - 1u : op.blocks;
    volatile uint32_t* save_gpr =
        reinterpret_cast<volatile uint32_t*>(op.scratch + WHS_SAVE_GPR + TT_TRISC_INDEX * 256u);
    for (uint32_t i = 0; i < 64u; ++i) {
        save_gpr[i] = wh_gpr(i);
    }
    volatile uint32_t* cfg = reinterpret_cast<volatile uint32_t*>(wh_tensix_cfg);
    volatile uint32_t* save_cfg = reinterpret_cast<volatile uint32_t*>(op.scratch + WHS_SAVE_CFG);
#if TT_TRISC_INDEX == 0
    for (uint32_t i = 0; i < CFG_STATE_SIZE * 4u; ++i) {
        save_cfg[i] = cfg[i];
    }
    sync_word(WHS_S_SAVED) = op.go;
#else
    while (sync_word(WHS_S_SAVED) != op.go) {
    }
#endif
    // drain()'s marks count from 1 through gpr_mark, which may hold an earlier program's last.
    wh_gpr(gpr_mark) = 0u;
    setup();
    wh_thread_idle(wh_wait_idle);
#if TT_TRISC_INDEX == 1
    sync_word(WHS_S_MATH_READY) = op.go;
#else
    while (sync_word(WHS_S_MATH_READY) != op.go) {
    }
#endif
    run();
    wh_thread_idle(wh_wait_idle);
    sync_word(WHS_S_DONE + TT_TRISC_INDEX) = op.go;
    for (uint32_t i = 0; i < 3u; ++i) {
        while (sync_word(WHS_S_DONE + i) != op.go) {
        }
    }
    cleanup();
    wh_thread_idle(wh_wait_idle);
    for (uint32_t i = 0; i < 64u; ++i) {
        wh_gpr(i) = save_gpr[i];
    }
#if TT_TRISC_INDEX == 0
    for (uint32_t i = 0; i < CFG_STATE_SIZE * 4u; ++i) {
        cfg[i] = save_cfg[i];
    }
    sync_word(WHS_S_RESTORED) = op.go;
#else
    while (sync_word(WHS_S_RESTORED) != op.go) {
    }
#endif
}
