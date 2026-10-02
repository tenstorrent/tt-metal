// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase A (prep) reader: constants (eye, tril, ones, masks) once, then per-chunk q,k,v,g,beta.
// No initial state — the prep phase is state-independent. Device 2.0 API.
// The reads of one item, and at kickoff the constants as well, are issued as one flight behind a single
// read barrier: the compute cannot start an item before q and k have landed, and at kickoff every
// producer of the fused program reads at once, so each extra barrier is a contended round trip.
//
// GDN_FUSED_PRODUCER + GDN_DYNAMIC_ITEMS (chunk_gdn_fused_map.hpp): the items are claimed at run time. A home producer
// (p < BH*NPH, head p / NPH, rank p % NPH) starts with chunk rank of its head and claims its head's later chunks; an
// extra claims from head x*BH/NX and moves one head on after each claim; a target head whose counter is past NC is
// exhausted and the producer moves on, until all BH heads are. A claim is a NoC fetch-and-add on head h's counter
// (receiver (h, 0); the response, the pre-increment value c, lands in this core's return word), issued right after the
// current item's reads and collected only after the next item's input slots are reserved, so its round trip hides
// behind an item. Per item: item[n % Q] = h << 16 | c, SEM_PUB = n + 1 (the writer and the compute spin on it), this
// core registered as the owner of (h, c) at the head's NV receivers (owner[c] = x | y << 8 | (n % Q) << 16 | valid
// bit, an inline dword write), then the reads. Once every head is exhausted SEM_FIN = n + 1 ends the loop. The first
// registration and claim wait for SEM_READY (the aggregating receiver's "every counter seeded, every owner table
// zeroed").

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#if defined(GDN_FUSED_PRODUCER)
#include "chunk_gdn_fused_map.hpp"
#endif
#if defined(GDN_FUSED_PRODUCER) && defined(GDN_DYNAMIC_ITEMS)
#include "api/dataflow/noc_semaphore.h"
#include "risc_common.h"
#endif

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7;
// Three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10) packed into one [1,1,32,96] tensor.
// Loaded once into the cb_u slot (17), which the stable-form prep no longer uses.
constexpr uint32_t cb_mask = 17;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);

    constexpr auto q_a = TensorAccessorArgs<3>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto eye_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto tril_a = TensorAccessorArgs<eye_a.next_compile_time_args_offset()>();
    constexpr auto ones_a = TensorAccessorArgs<tril_a.next_compile_time_args_offset()>();
    constexpr auto mask_a = TensorAccessorArgs<ones_a.next_compile_time_args_offset()>();
    // OPT-A: trailing compile args (after all TensorAccessorArgs). 1 => read that tensor FLAT token-major.
    constexpr uint32_t V_FLAT = get_compile_time_arg_val(mask_a.next_compile_time_args_offset());
    constexpr uint32_t QK_FLAT = get_compile_time_arg_val(mask_a.next_compile_time_args_offset() + 1);
#if defined(GDN_FUSED_PRODUCER) && defined(GDN_DYNAMIC_ITEMS)
    // Dynamic hand-off: the control-block CB (the u/mask CB) with the credit-tile and owner-table offsets, then the
    // semaphore ids: ready (the aggregating receiver -> producers), published / finished (local).
    constexpr uint32_t CB_CTRL = get_compile_time_arg_val(mask_a.next_compile_time_args_offset() + 2);
    constexpr uint32_t CTRL_OFF = get_compile_time_arg_val(mask_a.next_compile_time_args_offset() + 3);
    constexpr uint32_t OWNER_OFF = get_compile_time_arg_val(mask_a.next_compile_time_args_offset() + 4);
    constexpr uint32_t SEM_READY = get_compile_time_arg_val(mask_a.next_compile_time_args_offset() + 5);
    constexpr uint32_t SEM_PUB = get_compile_time_arg_val(mask_a.next_compile_time_args_offset() + 6);
    constexpr uint32_t SEM_FIN = get_compile_time_arg_val(mask_a.next_compile_time_args_offset() + 7);
#endif

    // A work-item is a flat (head, chunk) index — exactly the DRAM tile-group index h*NC + c.
#if defined(GDN_FUSED_PRODUCER) && defined(GDN_DYNAMIC_ITEMS)
    // Dynamic hand-off: producer p's role comes from the map (home producer or extra); arg 1 (item count) is unused.
    const uint32_t p = get_arg_val<uint32_t>(0);
#elif defined(GDN_FUSED_PRODUCER)
    // Fused producer p of the map (chunk_gdn_fused_map.hpp): its wi_count items in map order.
    const uint32_t p = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);
#else
    // Chunk-parallel: this core handles the work-items wi_start + n * wi_stride, n < wi_count.
    const uint32_t wi_start = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);
#endif
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);
    const uint32_t b_addr = get_arg_val<uint32_t>(6);
    const uint32_t eye_addr = get_arg_val<uint32_t>(7);
    const uint32_t tril_addr = get_arg_val<uint32_t>(8);
    const uint32_t ones_addr = get_arg_val<uint32_t>(9);
    const uint32_t mask_addr = get_arg_val<uint32_t>(10);
    // Flat metadata (used by V_FLAT/QK_FLAT): NC = chunks/head, HV = value-head count, Hk = key-head count.
    const uint32_t NC = get_arg_val<uint32_t>(11);
    const uint32_t HV = get_arg_val<uint32_t>(12);
    const uint32_t Hk = get_arg_val<uint32_t>(13);
    // Cycles to wait before the first read: the fused factory's kickoff stagger, 0 for the phased prep.
    const uint32_t kickoff_wait_cycles = get_arg_val<uint32_t>(15);
#if defined(GDN_FUSED_PRODUCER) && defined(GDN_DYNAMIC_ITEMS)
    // Dynamic hand-off: BH, NPH and NX (home producers per head, extras; args 18..19 of the map are unused), the
    // SEM_READY count, NV and this core's own packed coords. Common args: the fused writer's head table (per head the
    // rectangle word, then the NV receivers' coords two per word).
    const uint32_t BH = get_arg_val<uint32_t>(14);
    const uint32_t NPH = get_arg_val<uint32_t>(16);
    const uint32_t NX = get_arg_val<uint32_t>(17);
    const uint32_t ready_count = get_arg_val<uint32_t>(20);
    const uint32_t NV = get_arg_val<uint32_t>(21);
    const uint32_t my_xy = get_arg_val<uint32_t>(22);
#elif defined(GDN_FUSED_PRODUCER)
    // The producer map: BH, then (NPH, NX, num, den) as in chunk_gdn_fused_map.hpp.
    const GdnFusedMap map{
        get_arg_val<uint32_t>(14),
        NC,
        get_arg_val<uint32_t>(16),
        get_arg_val<uint32_t>(17),
        get_arg_val<uint32_t>(18),
        get_arg_val<uint32_t>(19)};
    auto item_wi = [&](uint32_t n) {
        const GdnFusedItem it = gdn_fused_item(map, p, n);
        return it.h * NC + it.c;
    };
#else
    // Work-item stride: 1 for a contiguous slice (phased prep).
    const uint32_t wi_stride = get_arg_val<uint32_t>(14);
    auto item_wi = [&](uint32_t n) { return wi_start + n * wi_stride; };
#endif

    // Mixed precision: q/k/v are bf16; g/beta and the constants are fp32.
    const uint32_t tb_io = get_tile_size(cb_q);
    const uint32_t tb_f = get_tile_size(cb_g);
    const auto q_acc = TensorAccessor(q_a, q_addr, tb_io);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb_io);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb_io);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb_f);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb_f);
    const auto eye_acc = TensorAccessor(eye_a, eye_addr, tb_f);
    const auto tril_acc = TensorAccessor(tril_a, tril_addr, tb_f);
    const auto ones_acc = TensorAccessor(ones_a, ones_addr, tb_f);
    const auto mask_acc = TensorAccessor(mask_a, mask_addr, tb_f);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;

    Noc noc;

    // Reserve `n` tiles of a CB and post their reads; no barrier, no push. The caller ends a flight with
    // `noc.async_read_barrier()` and pushes every CB it reserved.
    auto issue = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n, uint32_t tb) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
    };
    auto publish = [](uint32_t cb_id, uint32_t n) { CircularBuffer(cb_id).push_back(n); };

    // Flat-v token-major read: fetch head hv's chunk c out of the flat [B,T,HV*V] tile grid
    // (row stride HV*Vt tiles, column offset hv*Vt), packing the [Ct,Vt] block contiguously into
    // cb_v in the SAME row-major order the head-major read produces (CB idx rt*Vt+ct) — so the
    // compute sees byte-identical tiles regardless of source layout. Requires pad==0 (T=NC*Ct*32).
    auto issue_v_flat = [&](uint32_t hc) {
        const uint32_t bh = hc / NC;
        const uint32_t c = hc % NC;
        const uint32_t hv = bh % HV;
        const uint32_t b = bh / HV;
        const uint32_t row_stride = HV * Vt;                   // tiles per token-row in flat v
        const uint32_t batch_base = b * NC * Ct * row_stride;  // b * (T/32) * row_stride
        CircularBuffer cbv(cb_v);
        cbv.reserve_back(cv);
        for (uint32_t rt = 0; rt < Ct; rt++) {
            for (uint32_t ct = 0; ct < Vt; ct++) {
                const uint32_t page = batch_base + (c * Ct + rt) * row_stride + hv * Vt + ct;
                noc.async_read(v_acc, cbv, tb_io, {.page_id = page}, {.offset_bytes = (rt * Vt + ct) * tb_io});
            }
        }
    };

    // Flat-q/k token-major read: work-item is value-head hv; its key-head is hk = hv / G (GQA group
    // size G = HV/Hk). Fetch [Ct,Kt] for (hk, chunk c) from the flat [B,T,Hk*K] grid (row stride Hk*Kt,
    // col offset hk*Kt), packed row-major into `cb` — identical layout to the head-major read.
    auto issue_qk_flat = [&](const auto& acc, uint32_t cb_id, uint32_t hc) {
        const uint32_t G = HV / Hk;
        const uint32_t bh = hc / NC;
        const uint32_t c = hc % NC;
        const uint32_t hv = bh % HV;
        const uint32_t b = bh / HV;
        const uint32_t hk = hv / G;
        const uint32_t row_stride = Hk * Kt;
        const uint32_t batch_base = b * NC * Ct * row_stride;
        CircularBuffer cb(cb_id);
        cb.reserve_back(ck);
        for (uint32_t rt = 0; rt < Ct; rt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                const uint32_t page = batch_base + (c * Ct + rt) * row_stride + hk * Kt + kt;
                noc.async_read(acc, cb, tb_io, {.page_id = page}, {.offset_bytes = (rt * Kt + kt) * tb_io});
            }
        }
    };

    // One item's inputs, q and k first (the norm consumes them first), as one flight.
    auto issue_item = [&](uint32_t hc) {
        if constexpr (QK_FLAT) {
            issue_qk_flat(q_acc, cb_q, hc);
            issue_qk_flat(k_acc, cb_k, hc);
        } else {
            issue(q_acc, cb_q, hc * ck, ck, tb_io);
            issue(k_acc, cb_k, hc * ck, ck, tb_io);
        }
        if constexpr (V_FLAT) {
            issue_v_flat(hc);
        } else {
            issue(v_acc, cb_v, hc * cv, cv, tb_io);
        }
        issue(g_acc, cb_g, hc * Ct, Ct, tb_f);
        issue(b_acc, cb_beta, hc * Ct, Ct, tb_f);
    };
    auto publish_item = [&]() {
        publish(cb_q, ck);
        publish(cb_k, ck);
        publish(cb_v, cv);
        publish(cb_g, Ct);
        publish(cb_beta, Ct);
    };

#if defined(GDN_FUSED_PRODUCER) && defined(GDN_DYNAMIC_ITEMS)
    // Control block in this core's credit tile (chunk_gdn_fused_map.hpp offsets); SEM_PUB / SEM_FIN are semaphore
    // words, so dispatch resets them every launch. The owner tables sit at OWNER_OFF of every receiver's u/mask CB.
    const uint32_t u_base = CircularBuffer(CB_CTRL).get_read_ptr();
    const uint32_t ret_addr = u_base + CTRL_OFF + kGdnDynRetOff;
    volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ret_addr);
    volatile tt_l1_ptr uint32_t* items =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u_base + CTRL_OFF + kGdnDynItemOff);
    volatile tt_l1_ptr uint32_t* pub = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PUB));
    volatile tt_l1_ptr uint32_t* fin = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_FIN));
    // The compute kernel's copies: zeroed here, before the first claim and before the mask tiles it waits for.
    volatile tt_l1_ptr uint32_t* cpub =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u_base + CTRL_OFF + kGdnDynPubOff);
    volatile tt_l1_ptr uint32_t* cfin =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u_base + CTRL_OFF + kGdnDynFinOff);
    *cpub = 0;
    *cfin = 0;
    asm volatile("fence");
    const uint32_t owner_base = u_base + OWNER_OFF;
    const uint8_t nid = noc.get_noc_id();
    const uint32_t head_words = 1 + (NV + 1) / 2;
    auto rcv_word = [&](uint32_t h, uint32_t v) {
        return (get_common_arg_val<uint32_t>(h * head_words + 1 + v / 2) >> (16 * (v % 2))) & 0xFFFFu;
    };
    const uint32_t ctr_off = u_base + CTRL_OFF + kGdnDynHeadCtrOff;  // head h's counter: this offset on receiver (h, 0)

    struct Item {
        uint32_t h, c, idx;  // idx = n % Q: the credit word and item word of this item
    };
    uint32_t n = 0;  // items published
    // Publish (h, c) as this core's item n: the item word first, then SEM_PUB = n + 1 and the compute's copy.
    auto publish_item_word = [&](Item& it, uint32_t h, uint32_t c) {
        it = Item{h, c, n % kGdnDynQ};
        items[it.idx] = (h << 16) | c;
        asm volatile("fence");
        *pub = ++n;
        *cpub = n;
    };
    // The role: a home producer claims from its head until that is exhausted; an extra starts at a head of its own
    // (the extras spread evenly over the heads) and moves one head on after every claim; past an exhausted head both
    // move on, and once all BH heads are exhausted the items are over.
    const bool home = p < BH * NPH;
    uint32_t target = home ? p / NPH : ((p - BH * NPH) * BH / NX) % BH;
    uint32_t exhausted = 0;  // consecutive target heads found past NC
    // The claim: one fetch-and-add in flight at a time, issued by claim_issue on the target head's counter and
    // collected by claim_complete, which publishes the item (moving the target on when so required), or, with every
    // head exhausted, sets SEM_FIN = n + 1 and returns false.
    auto claim_issue = [&]() {
        const uint32_t w = rcv_word(target, 0);
        noc_fast_atomic_increment<DM_DEDICATED_NOC, /*program_ret_addr=*/true>(
            nid,
            write_at_cmd_buf,
            get_noc_addr(w & 0xFFu, w >> 8, ctr_off, nid),
            NOC_UNICAST_WRITE_VC,
            1,
            31,
            false,
            false,
            ret_addr);
    };
    auto claim_complete = [&](Item& it) -> bool {
        DeviceZoneScopedN("p_claim");
        for (;;) {
            noc.async_atomic_barrier();  // the pre-increment value is in ret[0]
            const uint32_t c = ret[0];
            if (c < NC) {
                publish_item_word(it, target, c);
                exhausted = 0;
                if (!home) {
                    target = (target + 1) % BH;
                }
                return true;
            }
            if (++exhausted == BH) {
                *fin = n + 1;
                *cfin = n + 1;
                return false;
            }
            target = (target + 1) % BH;
            claim_issue();
        }
    };
    // The input slots of one item, without the reads: the wait for the compute to pop an earlier item, taken before
    // the claim is collected so the claim's round trip hides behind it (issue_item's own reserves then return at once).
    auto reserve_item = [&]() {
        CircularBuffer(cb_q).reserve_back(ck);
        CircularBuffer(cb_k).reserve_back(ck);
        CircularBuffer(cb_v).reserve_back(cv);
        CircularBuffer(cb_g).reserve_back(Ct);
        CircularBuffer(cb_beta).reserve_back(Ct);
    };
    // Register this core as the owner of (h, c) at head h's receivers: owner[c] on each.
    auto register_owner = [&](const Item& it) {
        DeviceZoneScopedN("p_reg");
        const uint32_t word = my_xy | (it.idx << 16) | kGdnDynOwnerValid;
        for (uint32_t v = 0; v < NV; v++) {
            const uint32_t w = rcv_word(it.h, v);
            noc_inline_dw_write(get_noc_addr(w & 0xFFu, w >> 8, owner_base + 4 * it.c, nid), word, 0xF, nid);
        }
    };

    // Kickoff: a home producer's first item (chunk = its rank, below its head's seeded counter) and the constants in
    // one flight; the registration and the first claim wait for the receivers' seeded counters and zeroed owner
    // tables (SEM_READY).
    if (kickoff_wait_cycles != 0) {
        riscv_wait(kickoff_wait_cycles);
    }
    Item cur{};
    if (home) {
        publish_item_word(cur, target, p % NPH);
        issue_item(cur.h * NC + cur.c);
    }
    issue(eye_acc, cb_eye, 0, cc, tb_f);
    issue(tril_acc, cb_tril, 0, cc, tb_f);
    issue(ones_acc, cb_ones, 0, cc, tb_f);
    issue(mask_acc, cb_mask, 0, 3, tb_f);  // Qtl, Qbr, Q10 (tiles 0,1,2)
    Semaphore<>(SEM_READY).wait(ready_count);
    if (home) {
        register_owner(cur);
    }
    claim_issue();
    noc.async_read_barrier();
    if (home) {
        publish_item();
    }
    publish(cb_eye, cc);
    publish(cb_tril, cc);
    publish(cb_ones, cc);
    publish(cb_mask, 3);

    for (;;) {
        reserve_item();
        if (!claim_complete(cur)) {
            break;
        }
        register_owner(cur);
        issue_item(cur.h * NC + cur.c);
        claim_issue();  // collected after the next reserve: a whole item to return
        noc.async_read_barrier();
        publish_item();
    }
    // Drain the registrations (non-posted writes) and the claims (non-posted atomics), then give the AT command
    // buffer its default return address back: the claims pointed it at ret_addr and the firmware re-initialises it
    // only when the NoC mode changes.
    noc.async_write_barrier();
    noc.async_atomic_barrier();
    while (!noc_cmd_buf_ready(nid, write_at_cmd_buf));
    noc_cmd_buf_set_ret_addr(nid, write_at_cmd_buf, NOC_XY_ADDR(my_x[nid], my_y[nid], MEM_NOC_ATOMIC_RET_VAL_ADDR));
#else
    // Kickoff: the first item's inputs and the constants in one flight.
    if (kickoff_wait_cycles != 0) {
        riscv_wait(kickoff_wait_cycles);
    }
    if (wi_count > 0) {
        issue_item(item_wi(0));
    }
    issue(eye_acc, cb_eye, 0, cc, tb_f);
    issue(tril_acc, cb_tril, 0, cc, tb_f);
    issue(ones_acc, cb_ones, 0, cc, tb_f);
    issue(mask_acc, cb_mask, 0, 3, tb_f);  // Qtl, Qbr, Q10 (tiles 0,1,2)
    noc.async_read_barrier();
    if (wi_count > 0) {
        publish_item();
    }
    publish(cb_eye, cc);
    publish(cb_tril, cc);
    publish(cb_ones, cc);
    publish(cb_mask, 3);

    for (uint32_t i = 1; i < wi_count; i++) {
        issue_item(item_wi(i));
        noc.async_read_barrier();
        publish_item();
    }
#endif
}
