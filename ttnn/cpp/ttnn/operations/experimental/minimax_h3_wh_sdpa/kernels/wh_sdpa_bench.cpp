// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Single-core bench of the WH SDPA core loop (wh_sdpa_core.h) as a tt-metal compute kernel: one 64-query block of a
// head of 128 against `blocks` 64-key blocks resident in L1, as tt-lab-private's trisc_wh_sdpa.cpp. Operands are L1
// tensors laid out by the host in the core loop's chunk layouts (Q^T then two P^T buffers at q_base, key blocks at
// kv_base); the TRISCs pack O^T / l's two d halves and the records to out_base, and time themselves into `sync`.

#include <cstdint>
#include "api/compute/common.h"
#include "wh_sdpa_core.h"

struct WhSdpaSync {
    uint32_t saved;
    uint32_t pad[3];
    uint32_t ready[4];
    uint32_t done[4];
    uint32_t start[3][2];
    uint32_t end[3][2];
    uint32_t restored;
    uint32_t pad2[15];
    // The configuration (state 0) as the previous op left it: the core loop rewrites unpacker, packer and ALU
    // registers that tt-metal's LLK inits do not all set again, so TRISC0 puts every word back at the end.
    uint32_t cfg[CFG_STATE_SIZE * 4];
    // Each thread's 64 GPRs, which tt-metal's LLK keeps state in (ckernel_gpr_map.h).
    uint32_t gpr[3][64];
};

struct Args {
    uint32_t q_base, p_base, kv_base, out_base, blocks, last_valid, debug, sync, go;
};

static Args args() {
    Args a;
    a.q_base = get_arg_val<uint32_t>(0);
    a.p_base = a.q_base + WH_SDPA_Q_BYTES;
    a.kv_base = get_arg_val<uint32_t>(1);
    a.out_base = get_arg_val<uint32_t>(2);
    a.blocks = get_arg_val<uint32_t>(3);
    a.last_valid = get_arg_val<uint32_t>(4);
    a.debug = get_arg_val<uint32_t>(5);
    a.sync = get_arg_val<uint32_t>(6);
    a.go = get_arg_val<uint32_t>(7);
    return a;
}

#if TT_TRISC_INDEX == 0

static void setup(const Args& b) {
    setup_unpack<whp_sdpa_head128>(b.q_base, b.kv_base);
    wh_thread_idle(wh_wait_idle);
}

static void run(const Args& b) {
    unpack_qk<whp_sdpa_head128>(0u);
    for (uint32_t j = 0; j < b.blocks; ++j) {
        if (j + 1u < b.blocks) {
            unpack_qk<whp_sdpa_head128>(j + 1u);
        }
        unpack_pv<whp_sdpa_head128>(j % 2u, j);
    }
}

static void cleanup() { wh_thread_reset(); }

#elif TT_TRISC_INDEX == 1

static void setup(const Args&) {
    setup_math();
    wh_thread_idle(wh_wait_idle);
}

static void run(const Args& b) {
    constexpr uint32_t phases = 2u;  // HiFi2
    qk<whp_sdpa_head128>(0u, false, true, phases);
    for (uint32_t j = 0; j < b.blocks; ++j) {
        if (j + 1u < b.blocks) {
            qk<whp_sdpa_head128>((j + 1u) % 2u, true, false, phases);
        }
        pv<whp_sdpa_head128>(phases, false);
    }
}

static void cleanup() { wh_thread_reset(); }

#else

static void setup(const Args&) {
    setup_softmax();
    reset_records();
    wh_thread_idle(wh_wait_idle);
}

// The records into rows 0-7 of the S^T buffer at `base`: m in rows 0-3, l in rows 4-7, query face r in row r.
[[gnu::always_inline]] static inline void store_records(uint32_t base) {
    set_dst_offset(base);
#pragma GCC unroll 16
    for (uint32_t p = 0; p < 2u; ++p) {
        mov(6u + p, 0u);
        store(0u, 2u * p);
        shift(0u, 16);
        store(0u, 4u + 2u * p);
    }
}

static void run(const Args& b) {
    pack_format(true);
    for (uint32_t j = 0; j < b.blocks; ++j) {
        const uint32_t valid = j + 1u == b.blocks ? b.last_valid : 64u;
        const uint32_t p = b.p_base + j % 2u * WH_SDPA_REGION_BYTES;
        if (j % 2u) {
            block<whp_sdpa_head128, 1u>(p, true, valid, false, !(b.debug & 1u));
        } else {
            block<whp_sdpa_head128, 0u>(p, j != 0u, valid, false, !(b.debug & 1u));
        }
    }
    finish<whp_sdpa_head128>(b.blocks);
    store_records(s_base(0u));
    push(TT_OP_STALLWAIT(wh_block_pack | wh_block_sync | wh_block_cfg | wh_block_thcon, wh_wait_sfpu));
    pack_format(false);
    const uint32_t rows[3] = {o_base(0u), o_base(1u), s_base(0u)};
#pragma GCC unroll 16
    for (uint32_t r = 0; r < 3u; ++r) {
        wh_pack_region(gpr_out, b.out_base + r * WH_SDPA_REGION_BYTES, WH_SDPA_REGION_BYTES / 4u, rows[r] / 16u);
    }
}

static void cleanup() {
    push(TT_OP_STALLWAIT(wh_block_sync | wh_block_cfg | wh_block_thcon, wh_wait_pack));
    // LReg 11 is -1.0 by tt-metal's convention (set once at boot, c_tensix_core.h) and LLK code relies on it; the core
    // loop's broadcasts (park_faces) overwrite LRegs 11-14.
    push(TT_OP_SFPLOADI(0u, loadi_bf16, 0xBF80u));
    push(TT_OP_SFPCONFIG(0u, 11u, 0u));
    wh_thread_reset();
}

#endif

void kernel_main() {
    const Args b = args();
    volatile WhSdpaSync* s = reinterpret_cast<volatile WhSdpaSync*>(b.sync);
    volatile uint32_t* cfg = reinterpret_cast<volatile uint32_t*>(wh_tensix_cfg);
    for (uint32_t i = 0; i < 64u; ++i) {
        s->gpr[TT_TRISC_INDEX][i] = wh_gpr(i);
    }
#if TT_TRISC_INDEX == 0
    for (uint32_t i = 0; i < CFG_STATE_SIZE * 4u; ++i) {
        s->cfg[i] = cfg[i];
    }
    s->saved = b.go;
#else
    while (s->saved != b.go) {
    }
#endif
    // Debug bits: 256 skips setup and run, 512 runs setup only, 1024 << t skips thread t's setup.
    const bool do_setup = !(b.debug & 256u) && !(b.debug & (1024u << TT_TRISC_INDEX));
    const bool do_run = !(b.debug & (256u | 512u));
    if (do_setup) {
        setup(b);
    }
    s->ready[TT_TRISC_INDEX] = b.go;
    for (uint32_t i = 0; i < 3u; ++i) {
        while (s->ready[i] != b.go) {
        }
    }
    wh_put_time(s->start[TT_TRISC_INDEX]);
    if (do_run) {
        run(b);
    }
    wh_thread_idle(wh_wait_idle);
    wh_put_time(s->end[TT_TRISC_INDEX]);
    s->done[TT_TRISC_INDEX] = b.go;
    for (uint32_t i = 0; i < 3u; ++i) {
        while (s->done[i] != b.go) {
        }
    }
    cleanup();
    wh_thread_idle(wh_wait_idle);
    for (uint32_t i = 0; i < 64u; ++i) {
        wh_gpr(i) = s->gpr[TT_TRISC_INDEX][i];
    }
#if TT_TRISC_INDEX == 0
    for (uint32_t i = 0; i < CFG_STATE_SIZE * 4u; ++i) {
        cfg[i] = s->cfg[i];
    }
    s->restored = b.go;
#else
    while (s->restored != b.go) {
    }
#endif
}
