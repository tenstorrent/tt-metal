// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gated_delta_net_backward — reader (RISCV_1 / NCRISC).
//
// Owns every DRAM -> L1 load of the program, plus the two semaphore WAITS that
// separate the three stages.  Three sections:
//
//   Stage P : face-row gather of q/k/v/do (+ the g/beta column gather), the
//             padded-tail zero fill, and the head-major compact copy-out to the
//             `scin` scratch.  This is the dominant data-movement term of the op.
//   Stage S : per-chunk loads of the prep scratch for the sequential scan
//             (owner cores only), preceded by the Barrier-1 wait.
//   Stage G : full-page compact reads + the derived per-chunk / per-V-block
//             scratch, preceded by the Barrier-2 wait.
//
// THE FACE-ROW GATHER.  q is [B,T,H,K] in TILE layout, so a page is
// [32 heads x 32 key-dims] for ONE token and a single head is one ROW of every
// page.  A tile row lives in two faces, 16 elements each, exactly 256 elements
// apart, so one NoC read of a 272-element span covers the whole row.  The NoC's
// only alignment rule is that the L1 destination and the DRAM source agree
// modulo 64 (NOC_DRAM_READ_ALIGNMENT_BYTES), which is why each staging line
// starts at `stage64 + (src_byte_off & 63)`.  The two 16-element runs are then
// re-packed into the destination tile's face rows with plain RISC-V stores.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

#include "gdn_common.hpp"

// -Os: the three kernels of this op share ONE kernel-config ring buffer and the
// compute binaries dominate it, so every byte here counts.  These loops are
// NoC-issue bound, not instruction bound.
#pragma GCC optimize("Os")

// ---------------------------------------------------------------------------
// Constant-tile construction (once per core, at boot)
// ---------------------------------------------------------------------------
//
// Five [C,C] 0/1 mask blocks plus a bias block and a scalar tile.  The blocks
// are built with raw L1 stores because `ttnn.zeros` / `ttnn.eye` on the host
// would each be a second op dispatch, which this op's single-dispatch contract
// forbids.  Zero-filled first over the NoC (never with a CPU store loop), then
// only the non-zero lanes are written.

static __attribute__((noipa)) void store_f32(uint32_t addr, uint32_t elem, float value) {
    volatile tt_l1_ptr float* p = reinterpret_cast<volatile tt_l1_ptr float*>(addr);
    p[elem] = value;
}

// L1 address of the resident all-zeros constant tile (set once at boot).
static uint32_t g_zero_tile_addr = 0;
static constexpr uint32_t ZTILE_BYTES = 32 * 32 * 4;

// Zero an L1 range by NoC-copying the resident zero tile over it.  Never a CPU
// store loop: the DM engine moves the bytes and the RISC stays free.
static __attribute__((noipa)) void zero_l1(uint32_t dst, uint32_t nbytes) {
    while (nbytes) {
        const uint32_t chunk = nbytes > ZTILE_BYTES ? ZTILE_BYTES : nbytes;
        noc_async_write(g_zero_tile_addr, get_noc_addr(my_x[noc_index], my_y[noc_index], dst), chunk);
        dst += chunk;
        nbytes -= chunk;
    }
    noc_async_write_barrier();
}

void build_constant_tiles(uint32_t scale_bits) {
    constexpr uint32_t f32_tile = 32 * 32 * 4;
    constexpr uint32_t nconst = NCONST;

    cb_reserve_back(cb_const, nconst);
    const uint32_t cbase = get_write_ptr(cb_const);
    // Bootstrap: the zero tile itself is the only region written by the RISC.
    g_zero_tile_addr = cbase + CST_ZERO * f32_tile;
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_zero_tile_addr);
        for (uint32_t i = 0; i < 32 * 32; ++i) {
            z[i] = 0;
        }
    }
    zero_l1(cbase, CST_ZERO * f32_tile);
    zero_l1(cbase + (CST_ZERO + 1) * f32_tile, (nconst - CST_ZERO - 1) * f32_tile);

    for (uint32_t ti = 0; ti < Ct; ++ti) {
        for (uint32_t si = 0; si < Ct; ++si) {
            const uint32_t t = ti * Ct + si;
            const uint32_t lt = cbase + (CST_LT + t) * f32_tile;
            const uint32_t st = cbase + (CST_NSTRICT + t) * f32_tile;
            const uint32_t ut = cbase + (CST_UT + t) * f32_tile;
            const uint32_t ey = cbase + (CST_EYE + t) * f32_tile;
            const uint32_t bi = cbase + (CST_BIAS + t) * f32_tile;
            const uint32_t su = cbase + (CST_SUT + t) * f32_tile;
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t gr = ti * 32 + r;
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t gc = si * 32 + c;
                    const uint32_t e = tile_elem_off(r, c);
                    if (gc <= gr) {
                        store_f32(lt, e, 1.0f);
                    } else {
                        store_f32(bi, e, -1.0e4f);
                    }
                    if (gc < gr) {
                        // NSTRICT carries -1 so the two consumers of the strict
                        // mask (A and dA) come out already negated.
                        store_f32(st, e, -1.0f);
                    }
                    if (gc >= gr) {
                        store_f32(ut, e, 1.0f);
                    }
                    if (gc > gr) {
                        store_f32(su, e, 1.0f);
                    }
                    if (gc == gr) {
                        store_f32(ey, e, 1.0f);
                    }
                }
            }
        }
    }
    // Scale tile: column 0 only.  It is consumed as a stride-0 COL-broadcast
    // vector, which keeps the whole kernel to a single broadcast flavour (the
    // SCALAR-broadcast LLK instantiation costs more code than it saves).
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbase + CST_SCALE * f32_tile);
        for (uint32_t r = 0; r < 32; ++r) {
            p[tile_elem_off(r, 0)] = scale_bits;
        }
    }
    // Row-of-ones tiles: the [1,C] operand of the outer product that builds the
    // column-broadcast matrix for L (matmul instead of a unary broadcast op).
    for (uint32_t t = 0; t < Ct; ++t) {
        const uint32_t base = cbase + (CST_ROWONES + t) * f32_tile;
        for (uint32_t c = 0; c < 32; ++c) {
            store_f32(base, tile_elem_off(0, c), 1.0f);
        }
    }
    cb_push_back(cb_const, nconst);

    // Column-of-ones tiles: the row-sum matmul operand (X @ colones == rowsum).
    constexpr uint32_t ncol = NCOL;
    cb_reserve_back(cb_colones, ncol);
    const uint32_t colbase = get_write_ptr(cb_colones);
    zero_l1(colbase, ncol * f32_tile);
    for (uint32_t t = 0; t < ncol; ++t) {
        const uint32_t base = colbase + t * f32_tile;
        for (uint32_t r = 0; r < 32; ++r) {
            store_f32(base, tile_elem_off(r, 0), 1.0f);
        }
    }
    cb_push_back(cb_colones, ncol);
}

// ---------------------------------------------------------------------------
// Face-row gather
// ---------------------------------------------------------------------------

template <typename ACC>
void gather_block(
    const ACC& acc,
    uint32_t cb,
    uint32_t tile_bytes,
    uint32_t b,
    uint32_t t0,
    uint32_t dtot,
    uint32_t d0,
    uint32_t dn,
    uint32_t h,
    uint32_t block_pages) {
    cb_reserve_back(cb, block_pages);
    const uint32_t dst_base = get_write_ptr(cb);
    // Unconditional: the padded-tail contract needs rows t >= T to read as
    // zero, and the tail of a uniform push size larger than Ct*dn must not
    // carry the previous item's values.  Skipping the fill on whole chunks
    // (where the re-pack below overwrites every row anyway) was MEASURED and is
    // neutral-to-1.03x-WORSE: this reader is RISC-issue bound, not bandwidth
    // bound, so the DM engine's zeroing is free and the added branch is not.
    zero_l1(dst_base, block_pages * tile_bytes);

    const uint32_t src_off = row_run0_bytes(h);
    const uint32_t stage64 = (get_write_ptr(cb_gather) + 63u) & ~63u;
    const uint32_t line_shift = src_off & 63u;
    // A staging window is GATHER_TOKENS rows of ONE destination tile, so the
    // window count is (tile rows) x (windows per tile) x (d-tiles).  Splitting a
    // tile row into several windows is what lets GATHER_DEPTH slots be resident
    // without growing the staging buffer: the two knobs trade off exactly.
    constexpr uint32_t WPT = 32u / GATHER_TOKENS;
    const uint32_t nwin = Ct * WPT * dn;

    // Software-pipelined: window w's DRAM reads are in flight while window w-1's
    // two 16-element runs are re-packed by the RISC.  One transaction id per
    // staging slot is what makes that legal -- a plain noc_async_read_barrier()
    // would also wait for the reads just issued.
    // One in-flight count per staging slot; sized from the depth knob, never
    // from a literal, so raising GATHER_DEPTH cannot silently overrun it.
    uint32_t issued[GATHER_DEPTH] = {0};
    for (uint32_t w = 0; w <= nwin; ++w) {
        if (w < nwin) {
            const uint32_t j = w % dn;
            const uint32_t sub = (w / dn) % WPT;
            const uint32_t ct = w / (dn * WPT);
            const uint32_t slot = w % GATHER_DEPTH;
            const uint32_t sbase = stage64 + slot * GATHER_SLOT_BYTES;
            uint32_t n = 0;
            noc_async_read_set_trid(1 + slot);
            for (uint32_t r = 0; r < GATHER_TOKENS; ++r) {
                const uint32_t t = t0 + ct * 32 + sub * GATHER_TOKENS + r;
                if (t >= gT) {
                    break;
                }
                const uint32_t page = (b * gT + t) * dtot + (d0 + j);
                noc_async_read(
                    acc.get_noc_addr(page, src_off), sbase + r * ROW_SPAN_STRIDE + line_shift, ROW_SPAN_BYTES);
                ++n;
            }
            issued[slot] = n;
        }
        if (w > 0) {
            const uint32_t pw = w - 1;
            const uint32_t pj = pw % dn;
            const uint32_t psub = (pw / dn) % WPT;
            const uint32_t pct = pw / (dn * WPT);
            const uint32_t pslot = pw % GATHER_DEPTH;
            const uint32_t psbase = stage64 + pslot * GATHER_SLOT_BYTES;
            noc_async_read_barrier_with_trid(1 + pslot);
            const uint32_t tile_addr = dst_base + (pct * dn + pj) * tile_bytes;
            for (uint32_t r = 0; r < issued[pslot]; ++r) {
                const uint32_t src = psbase + r * ROW_SPAN_STRIDE + line_shift;
                const uint32_t dst = tile_addr + row_run0_bytes(psub * GATHER_TOKENS + r);
                copy_row_run(dst, src);
                copy_row_run(dst + ROW_RUN_GAP_BYTES, src + ROW_RUN_GAP_BYTES);
            }
        }
    }
    noc_async_read_set_trid(0);
}

// Column gather for the rank-3 gate tensors: head `h` is COLUMN h of page
// (b, t/32).  One whole-page read per token tile, then 32 scalar extracts.
template <typename ACC>
void gather_gate(
    const ACC& acc, uint32_t tile_bytes, uint32_t slot, uint32_t dst_base, uint32_t b, uint32_t t0, uint32_t h) {
    const uint32_t stage64 = (get_write_ptr(cb_gather) + 63u) & ~63u;
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        const uint32_t tbase = t0 + ct * 32;
        if (tbase >= gT) {
            break;
        }
        const uint32_t tt = tbase >> 5;
        noc_async_read(acc.get_noc_addr(b * Tt + tt, 0), stage64, tile_bytes);
        noc_async_read_barrier();
        const uint32_t tile_addr = dst_base + (slot + ct) * tile_bytes;
        for (uint32_t r = 0; r < 32; ++r) {
            if (tbase + r >= gT) {
                break;
            }
            const uint32_t se = tile_elem_off(r, h);
            const uint32_t de = tile_elem_off(r, 0);
            if constexpr (ESZ == 4) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage64);
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
                d[de] = s[se];
            } else {
                volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage64);
                volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
                d[de] = s[se];
            }
        }
    }
}

// Copy a freshly gathered block out to the head-major compact scratch.
template <typename ACC>
void compact_store(const ACC& acc, uint32_t src_base, uint32_t tile_bytes, uint32_t base_tile, uint32_t ntiles) {
    for (uint32_t i = 0; i < ntiles; ++i) {
        noc_async_write(src_base + i * tile_bytes, acc.get_noc_addr(base_tile + i), tile_bytes);
    }
    noc_async_write_barrier();
}

// Read a rows x cols sub-block out of the flat scratch buffer.
template <typename ACC>
static __attribute__((noipa)) void read_sub(
    const ACC& acc,
    uint32_t dst_addr,
    uint32_t base_tile,
    uint32_t rows,
    uint32_t row_stride,
    uint32_t col0,
    uint32_t cols,
    uint32_t tile_bytes) {
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_read(
                acc.get_noc_addr(base_tile + r * row_stride + col0 + c),
                dst_addr + (r * cols + c) * tile_bytes,
                tile_bytes);
        }
    }
}

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t g_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t do_addr = get_arg_val<uint32_t>(5);
    const uint32_t h0_addr = get_arg_val<uint32_t>(6);
    const uint32_t dht_addr = get_arg_val<uint32_t>(7);
    const uint32_t sc_addr = get_arg_val<uint32_t>(8);
    const uint32_t scin_addr = get_arg_val<uint32_t>(9);
    const uint32_t item_start = get_arg_val<uint32_t>(10);
    const uint32_t num_items = get_arg_val<uint32_t>(11);
    const uint32_t num_release_groups = get_arg_val<uint32_t>(13);
    const uint32_t scale_bits = get_arg_val<uint32_t>(14);
    const uint32_t num_owned = get_arg_val<uint32_t>(15);

    constexpr uint32_t in_tile = 32 * 32 * ESZ;
    constexpr uint32_t f32_tile = 32 * 32 * 4;

    constexpr auto qa = TensorAccessorArgs<CT_ACC_BASE>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto doa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto h0a = TensorAccessorArgs<doa.next_compile_time_args_offset()>();
    constexpr auto dhta = TensorAccessorArgs<h0a.next_compile_time_args_offset()>();
    constexpr auto sca = TensorAccessorArgs<dhta.next_compile_time_args_offset()>();
    constexpr auto scina = TensorAccessorArgs<sca.next_compile_time_args_offset()>();

    const auto q_acc = TensorAccessor(qa, q_addr, in_tile);
    const auto k_acc = TensorAccessor(ka, k_addr, in_tile);
    const auto v_acc = TensorAccessor(va, v_addr, in_tile);
    const auto g_acc = TensorAccessor(ga, g_addr, in_tile);
    const auto b_acc = TensorAccessor(ba, beta_addr, in_tile);
    const auto do_acc = TensorAccessor(doa, do_addr, in_tile);
    [[maybe_unused]] const auto h0_acc = TensorAccessor(h0a, h0_addr, in_tile);
    [[maybe_unused]] const auto dht_acc = TensorAccessor(dhta, dht_addr, in_tile);
    const auto sc_acc = TensorAccessor(sca, sc_addr, f32_tile);
    const auto scin_acc = TensorAccessor(scina, scin_addr, in_tile);

    build_constant_tiles(scale_bits);

    // ------------------------------------------------------------------
    // Stage P — prep.  One (bh, chunk) item per iteration.
    // ------------------------------------------------------------------
    for (uint32_t p = 0; p < num_items; ++p) {
        const uint32_t wi = item_start + p;
        const uint32_t bh = wi / NC;
        const uint32_t ci = wi % NC;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;
        const uint32_t t0 = ci * CHUNK;

        gather_block(q_acc, cb_qin, in_tile, b, t0, Kt, 0, Kt, h, CtKt);
        compact_store(scin_acc, get_write_ptr(cb_qin), in_tile, SI_Q + wi * CtKt, CtKt);
        cb_push_back(cb_qin, CtKt);

        gather_block(k_acc, cb_kin, in_tile, b, t0, Kt, 0, Kt, h, CtKt);
        compact_store(scin_acc, get_write_ptr(cb_kin), in_tile, SI_K + wi * CtKt, CtKt);
        cb_push_back(cb_kin, CtKt);

        // g and beta column tiles
        cb_reserve_back(cb_gatein, 2 * Ct);
        {
            const uint32_t gbase = get_write_ptr(cb_gatein);
            zero_l1(gbase, 2 * Ct * in_tile);
            gather_gate(g_acc, in_tile, 0, gbase, b, t0, h);
            gather_gate(b_acc, in_tile, Ct, gbase, b, t0, h);
        }
        cb_push_back(cb_gatein, 2 * Ct);

        for (uint32_t vb = 0; vb < NVB; ++vb) {
            gather_block(v_acc, cb_vin, in_tile, b, t0, Vt, vb * Vb, Vb, h, MAXV);
            {
                const uint32_t src = get_write_ptr(cb_vin);
                for (uint32_t r = 0; r < Ct; ++r) {
                    compact_store(scin_acc, src + r * Vb * in_tile, in_tile, SI_V + wi * CtVt + r * Vt + vb * Vb, Vb);
                }
            }
            cb_push_back(cb_vin, MAXV);

            gather_block(do_acc, cb_doin, in_tile, b, t0, Vt, vb * Vb, Vb, h, CtVb);
            {
                const uint32_t src = get_write_ptr(cb_doin);
                for (uint32_t r = 0; r < Ct; ++r) {
                    compact_store(scin_acc, src + r * Vb * in_tile, in_tile, SI_DO + wi * CtVt + r * Vt + vb * Vb, Vb);
                }
            }
            cb_push_back(cb_doin, CtVb);
        }
    }

    // ------------------------------------------------------------------
    // Barrier 1 — per-(bh) group fan-in.  ALL prep is issued above before ANY
    // wait here, which is what makes the rendezvous deadlock-free.
    // ------------------------------------------------------------------
    if (num_owned > 0) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PREP));
        noc_semaphore_wait_min(sem, NC * num_owned);

        // -------------------------------------------------------------
        // Stage S — the sequential scan, one core per (b,h).
        // -------------------------------------------------------------
        for (uint32_t oi = 0; oi < num_owned; ++oi) {
            const uint32_t bh = get_arg_val<uint32_t>(16 + oi);
            const uint32_t b = bh / gH;
            const uint32_t h = bh % gH;

            for (uint32_t vb = 0; vb < NVB; ++vb) {
                if constexpr (HAS_H0) {
                    cb_reserve_back(cb_vin, MAXV);
                    const uint32_t dst = get_write_ptr(cb_vin);
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        for (uint32_t j = 0; j < Vb; ++j) {
                            noc_async_read(
                                h0_acc.get_noc_addr(((b * gH + h) * Kt + kt) * Vt + vb * Vb + j),
                                dst + (kt * Vb + j) * in_tile,
                                in_tile);
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_vin, MAXV);
                }

                for (uint32_t i = 0; i < NC; ++i) {
                    const uint32_t wi = bh * NC + i;
                    cb_reserve_back(cb_load_vb, LVB);
                    uint32_t dst = get_write_ptr(cb_load_vb);
                    read_sub(sc_acc, dst, SC_KCD + wi * ST_KCD, Ct, Kt, 0, Kt, f32_tile);
                    dst += CtKt * f32_tile;
                    read_sub(sc_acc, dst, SC_P + wi * ST_P, Ct, Kt, 0, Kt, f32_tile);
                    dst += CtKt * f32_tile;
                    read_sub(sc_acc, dst, SC_VCORR + wi * ST_VCORR, Ct, Vt, vb * Vb, Vb, f32_tile);
                    dst += CtVb * f32_tile;
                    noc_async_read(sc_acc.get_noc_addr(SC_VEC + wi * ST_VEC + 2 * Ct), dst, f32_tile);
                    noc_async_read_barrier();
                    cb_push_back(cb_load_vb, LVB);
                }

                if constexpr (HAS_DHT) {
                    cb_reserve_back(cb_vin, MAXV);
                    const uint32_t dst = get_write_ptr(cb_vin);
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        for (uint32_t j = 0; j < Vb; ++j) {
                            noc_async_read(
                                dht_acc.get_noc_addr(((b * gH + h) * Kt + kt) * Vt + vb * Vb + j),
                                dst + (kt * Vb + j) * in_tile,
                                in_tile);
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_vin, MAXV);
                }

                for (int32_t i = (int32_t)NC - 1; i >= 0; --i) {
                    const uint32_t wi = bh * NC + (uint32_t)i;
                    cb_reserve_back(cb_load_vb, LVB);
                    uint32_t dst = get_write_ptr(cb_load_vb);
                    read_sub(sc_acc, dst, SC_P + wi * ST_P, Ct, Kt, 0, Kt, f32_tile);
                    dst += CtKt * f32_tile;
                    read_sub(sc_acc, dst, SC_U + wi * ST_U, Ct, Vt, vb * Vb, Vb, f32_tile);
                    dst += CtVb * f32_tile;
                    read_sub(sc_acc, dst, SC_KCD + wi * ST_KCD, Ct, Kt, 0, Kt, f32_tile);
                    dst += CtKt * f32_tile;
                    read_sub(sc_acc, dst, SC_C + wi * ST_C, Kt, Vt, vb * Vb, Vb, f32_tile);
                    dst += KtVb * f32_tile;
                    noc_async_read(sc_acc.get_noc_addr(SC_VEC + wi * ST_VEC + 2 * Ct), dst, f32_tile);
                    noc_async_read_barrier();
                    cb_push_back(cb_load_vb, LVB);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Barrier 2 — per-group release.  Cores with no stage-G work wait for 0.
    // ------------------------------------------------------------------
    {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_SCAN));
        noc_semaphore_wait_min(sem, num_release_groups);
    }

    // ------------------------------------------------------------------
    // Stage G — gradient assembly.  Full-page compact reads only.
    // ------------------------------------------------------------------
    for (uint32_t p = 0; p < num_items; ++p) {
        const uint32_t wi = item_start + p;

        cb_reserve_back(cb_qin, CtKt);
        {
            const uint32_t dst = get_write_ptr(cb_qin);
            for (uint32_t i = 0; i < CtKt; ++i) {
                noc_async_read(scin_acc.get_noc_addr(SI_Q + wi * CtKt + i), dst + i * in_tile, in_tile);
            }
        }
        cb_reserve_back(cb_kin, CtKt);
        {
            const uint32_t dst = get_write_ptr(cb_kin);
            for (uint32_t i = 0; i < CtKt; ++i) {
                noc_async_read(scin_acc.get_noc_addr(SI_K + wi * CtKt + i), dst + i * in_tile, in_tile);
            }
        }
        cb_reserve_back(cb_load_item, LITEM);
        {
            uint32_t dst = get_write_ptr(cb_load_item);
            for (uint32_t i = 0; i < CtCt; ++i) {
                noc_async_read(sc_acc.get_noc_addr(SC_ATTN + wi * ST_ATTN + i), dst + i * f32_tile, f32_tile);
            }
            dst += CtCt * f32_tile;
            for (uint32_t i = 0; i < 4 * Ct; ++i) {
                noc_async_read(sc_acc.get_noc_addr(SC_VEC + wi * ST_VEC + i), dst + i * f32_tile, f32_tile);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_qin, CtKt);
        cb_push_back(cb_kin, CtKt);
        cb_push_back(cb_load_item, LITEM);

        for (uint32_t vb = 0; vb < NVB; ++vb) {
            cb_reserve_back(cb_vin, MAXV);
            {
                const uint32_t dst = get_write_ptr(cb_vin);
                for (uint32_t r = 0; r < Ct; ++r) {
                    for (uint32_t j = 0; j < Vb; ++j) {
                        noc_async_read(
                            scin_acc.get_noc_addr(SI_V + wi * CtVt + r * Vt + vb * Vb + j),
                            dst + (r * Vb + j) * in_tile,
                            in_tile);
                    }
                }
            }
            cb_reserve_back(cb_doin, CtVb);
            {
                const uint32_t dst = get_write_ptr(cb_doin);
                for (uint32_t r = 0; r < Ct; ++r) {
                    for (uint32_t j = 0; j < Vb; ++j) {
                        noc_async_read(
                            scin_acc.get_noc_addr(SI_DO + wi * CtVt + r * Vt + vb * Vb + j),
                            dst + (r * Vb + j) * in_tile,
                            in_tile);
                    }
                }
            }
            cb_reserve_back(cb_load_vb, LVB);
            {
                uint32_t dst = get_write_ptr(cb_load_vb);
                read_sub(sc_acc, dst, SC_VNEW + wi * ST_VNEW, Ct, Vt, vb * Vb, Vb, f32_tile);
                dst += CtVb * f32_tile;
                read_sub(sc_acc, dst, SC_DVNEW + wi * ST_DVNEW, Ct, Vt, vb * Vb, Vb, f32_tile);
                dst += CtVb * f32_tile;
                read_sub(sc_acc, dst, SC_S + wi * ST_S, Kt, Vt, vb * Vb, Vb, f32_tile);
                dst += KtVb * f32_tile;
                read_sub(sc_acc, dst, SC_DS + wi * ST_DS, Kt, Vt, vb * Vb, Vb, f32_tile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_vin, MAXV);
            cb_push_back(cb_doin, CtVb);
            cb_push_back(cb_load_vb, LVB);
        }
    }
}
