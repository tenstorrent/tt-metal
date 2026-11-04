// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute-core reader (NCRISC, NoC0): load_q_block, await_landing, load_kv_block, release_ports.
//
// Per unit: the Q block (valid rows only; the CB credit is the nominal Sq*DHt so every block starts on the same
// slot), then every key segment (local -> joint -> remote by hop distance) in blocks of Sk key tiles. K tile (t, d)
// lands at CB index d*Sk + t (transposed block order for the Q.K^T matmul), V tile (t, d) at t*DVt + d; a ragged
// last block reads only its Sk_valid key tiles but pushes the nominal credit (compute reads only Sk_valid).
// Remote segments wait once per (unit, segment) for the landing counters of every port carrying a share of that
// source (OVERLAP: the local and joint blocks run before the first wait). At the end: +1 consumers_done on every
// port of this device, so the ports may re-arm their counters.
//
// K/V row multicast (design regime R2, host plan_kv_mcast_groups / plan_work_split): when the host attached a
// `kv_mcast` family, the cores of a row group hold, per pass (whole unit i < W), consecutive q blocks of ONE
// (batch, head group), so they walk the identical key-block stream. The group's injector (SenderPipe face) alone
// awaits the landings and reads each K/V block from DRAM, then multicasts the K block and the V block into the same
// cb_k / cb_v slots of every member (members push the same blocks in the same order, so the CB write pointers agree
// across the group); members (ReceiverPipe face) reserve the slots, ack, wait for the data and push. Cores in no
// group, and every core's tail piece, read their own K/V (R1).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ring_joint_sdpa_common.hpp"

// Sampled per-block zone window (key blocks of the first unit), matched to the compute kernel's.
constexpr uint32_t ZONE_SAMPLE_KB0 = 4;
constexpr uint32_t ZONE_SAMPLE_NKB = 6;

constexpr uint32_t MAX_PORTS = 16;

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t Sq = get_compile_time_arg_val(3);
    constexpr uint32_t Sk = get_compile_time_arg_val(4);
    constexpr uint32_t DHt = get_compile_time_arg_val(5);
    constexpr uint32_t DVt = get_compile_time_arg_val(6);
    constexpr uint32_t NQ = get_compile_time_arg_val(7);  // q tile-rows of a unit's row space (g heads stacked)
    constexpr uint32_t LQ = get_compile_time_arg_val(8);  // joint q tile-rows of a unit's row space
    constexpr uint32_t n_kv_t = get_compile_time_arg_val(9);
    constexpr uint32_t BG = get_compile_time_arg_val(10);  // b * hq / g (batch, head group) pairs
    [[maybe_unused]] constexpr uint32_t C = get_compile_time_arg_val(11);
    constexpr uint32_t hq = get_compile_time_arg_val(12);
    constexpr uint32_t hk = get_compile_time_arg_val(13);
    constexpr uint32_t hv = get_compile_time_arg_val(14);
    constexpr uint32_t g = get_compile_time_arg_val(15);  // q_heads_per_unit
    constexpr uint32_t n_q_t = get_compile_time_arg_val(16);
    constexpr uint32_t Lq_t = get_compile_time_arg_val(17);
    constexpr bool causal = get_compile_time_arg_val(18) != 0;
    constexpr bool balanced = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t G = get_compile_time_arg_val(20);
    constexpr uint32_t num_ports = get_compile_time_arg_val(21);
    constexpr uint32_t num_kinds = get_compile_time_arg_val(22);  // transport entry kinds: 1, 2 with sharded joint
    constexpr bool has_joint = Lq_t > 0;
    constexpr uint32_t groups = hq / g;
    constexpr uint32_t Hmax = hk > hv ? hk : hv;  // transport entries per batch (one per kv head index)
    static_assert(num_ports <= MAX_PORTS, "too many port cores");

    constexpr auto q_args = TensorAccessorArgs<23>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto jq_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto jk_args = TensorAccessorArgs<jq_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto jv_args = TensorAccessorArgs<jk_args.next_compile_time_args_offset()>();
    constexpr auto lk_args = TensorAccessorArgs<jv_args.next_compile_time_args_offset()>();
    constexpr auto lv_args = TensorAccessorArgs<lk_args.next_compile_time_args_offset()>();
    // sharded-joint landing (aliases of lk / lv otherwise: valid, never addressed)
    constexpr auto ljk_args = TensorAccessorArgs<lv_args.next_compile_time_args_offset()>();
    constexpr auto ljv_args = TensorAccessorArgs<ljk_args.next_compile_time_args_offset()>();
    constexpr auto kv_mc = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("kv_mcast_ct_offset"),
        get_named_compile_time_arg_val("kv_mcast_rt_offset")>();

    uint32_t a = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(a++);
    const uint32_t k_addr = get_arg_val<uint32_t>(a++);
    const uint32_t v_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t jq_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t jk_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t jv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lk_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ljk_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ljv_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t core_idx = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cdone_addr = get_arg_val<uint32_t>(a++);
    // Private L1 word the remote counter is read into. Never a counter word: placement changes between calls, so a
    // core polling here may be a port core next call, where a stale arrival count would read as landed items.
    const uint32_t poll_addr = get_arg_val<uint32_t>(a++);
    uint32_t port_x[MAX_PORTS], port_y[MAX_PORTS], seen[MAX_PORTS];
    for (uint32_t i = 0; i < num_ports; ++i) {
        port_x[i] = get_arg_val<uint32_t>(a + i);
        port_y[i] = get_arg_val<uint32_t>(a + num_ports + i);
        seen[i] = 0;
    }
    a += 2 * num_ports;
    const uint32_t num_segments = get_arg_val<uint32_t>(a++);
    const uint32_t seg_idx = a;  // per segment: kind, valid_tiles, masked, slot, S_in, rank, port_mask, src

    const uint32_t q_tile = get_tile_size(cb_q);
    const uint32_t kv_tile = get_tile_size(cb_k);
    const auto q_acc = TensorAccessor(q_args, q_addr, q_tile);
    const auto k_acc = TensorAccessor(k_args, k_addr, kv_tile);
    const auto v_acc = TensorAccessor(v_args, v_addr, kv_tile);
    const auto lk_acc = TensorAccessor(lk_args, lk_addr, kv_tile);
    const auto lv_acc = TensorAccessor(lv_args, lv_addr, kv_tile);
    const auto ljk_acc = TensorAccessor(ljk_args, ljk_addr, kv_tile);
    const auto ljv_acc = TensorAccessor(ljv_args, ljv_addr, kv_tile);

    // landing slot pages: slot * (b*hk*n_kv_t*DHt) + K head pages (V: hv, DVt)
    constexpr uint32_t B = BG / groups;
    constexpr uint32_t k_slot_pages = B * hk * n_kv_t * DHt;
    constexpr uint32_t v_slot_pages = B * hv * n_kv_t * DVt;
    constexpr uint32_t jk_slot_pages = B * hk * Lq_t * DHt;  // sharded joint: Lq_t = the slice's tiles
    constexpr uint32_t jv_slot_pages = B * hv * Lq_t * DVt;
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll_addr);

    const uint32_t tail_idx = seg_idx + num_segments * 8;  // W, then has_tail, bh, row0, rows, joint, then p
    const uint32_t W = get_arg_val<uint32_t>(tail_idx);
    [[maybe_unused]] const uint32_t my_pos = get_arg_val<uint32_t>(tail_idx + 6);  // group position p
    const uint32_t my_units = W + get_arg_val<uint32_t>(tail_idx + 1);
    const uint32_t units_idx = tail_idx + 7;  // the W whole-unit indices of this core, one per pass

    // ---- K/V row multicast roles (whole units only) ----
    Noc noc;
    auto kv_sender = kv_mc.optional_sender(noc);
    auto kv_receiver = kv_sender ? decltype(kv_mc.optional_receiver(noc)){} : kv_mc.optional_receiver(noc);
    const uint32_t k_block_bytes = Sk * DHt * kv_tile;
    const uint32_t v_block_bytes = Sk * DVt * kv_tile;

    for (uint32_t i = 0; i < my_units; ++i) {
        MaybeDeviceZoneScope("reader_unit");
        uint32_t kb = 0;  // key blocks issued for this unit (zone sampling)
        const AttnUnit unit =
            i < W ? decode_unit<Sq, NQ, LQ>(get_arg_val<uint32_t>(units_idx + i)) : tail_piece(tail_idx + 1);
        const bool mc_send = i < W && kv_sender.has_value();    // injector: read + multicast
        const bool mc_recv = i < W && kv_receiver.has_value();  // member: the injector's blocks land here
        const uint32_t b = unit.bh / groups;
        const uint32_t h = (unit.bh - b * groups) * g;  // first q head of the unit (all g share the K/V heads)
        const uint32_t kh = h / (hq / hk), vh = h / (hq / hv);
        const uint32_t k_head = b * hk + kh, v_head = b * hv + vh;
        // transport entry (b, j) carries K head j (j < hk) and V head j (j < hv): the unit needs both landed
        const uint32_t entry = b * Hmax + (kh > vh ? kh : vh);
        [[maybe_unused]] uint32_t max_gq = 0;
        if constexpr (causal) {
            max_gq = unit_max_gtile<balanced>(unit, my_pos, n_q_t, G);
        }

        // ---- load_q_block ----
        {
            MaybeDeviceZoneScope("reader_q_reserve");
            cb_reserve_back(cb_q, Sq * DHt);
        }
        {
            MaybeDeviceZoneScope("reader_q_read");
            uint32_t wptr = get_write_ptr(cb_q);
            if (!unit.joint) {
                uint32_t page = (unit.bh * NQ + unit.row0) * DHt;
                for (uint32_t t = 0; t < unit.rows * DHt; ++t) {
                    noc_async_read_tile(page + t, q_acc, wptr);
                    wptr += q_tile;
                }
            } else {
                if constexpr (has_joint) {
                    const auto jq_acc = TensorAccessor(jq_args, jq_addr, q_tile);
                    uint32_t page = (unit.bh * LQ + unit.row0) * DHt;
                    for (uint32_t t = 0; t < unit.rows * DHt; ++t) {
                        noc_async_read_tile(page + t, jq_acc, wptr);
                        wptr += q_tile;
                    }
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_q, Sq * DHt);

        // ---- key walk ----
        for (uint32_t s = 0; s < num_segments; ++s) {
            const uint32_t base = seg_idx + s * 8;
            const uint32_t kind = get_arg_val<uint32_t>(base + 0);
            uint32_t valid = get_arg_val<uint32_t>(base + 1);
            const uint32_t slot = get_arg_val<uint32_t>(base + 3);
            if constexpr (causal) {  // keys past the unit's latest query are never read (shared plan)
                valid = causal_visible_tiles<balanced>(valid, get_arg_val<uint32_t>(base + 7), max_gq, n_kv_t, G);
                if (valid == 0) {
                    continue;  // wholly in the future: no landing wait, no read
                }
            }
            uint32_t k_base, v_base;  // first page of this head in the source tensor
            if (kind == 1) {
                k_base = k_head * Lq_t * DHt;
                v_base = v_head * Lq_t * DVt;
            } else if (kind == 3) {
                k_base = slot * jk_slot_pages + k_head * Lq_t * DHt;
                v_base = slot * jv_slot_pages + v_head * Lq_t * DVt;
            } else {
                k_base = (kind == 2 ? slot * k_slot_pages : 0) + k_head * n_kv_t * DHt;
                v_base = (kind == 2 ? slot * v_slot_pages : 0) + v_head * n_kv_t * DVt;
            }
            if (kind >= 2 && !mc_recv) {
                MaybeDeviceZoneScope("reader_landing_wait");
                // ---- await_landing: count needed = ee * S_in + rank + 1 on every port carrying the source, ee the
                // entry index over (entry, kind): main slice ee = entry * num_kinds, joint slice + 1 ----
                const uint32_t S_in = get_arg_val<uint32_t>(base + 4);
                const uint32_t rank = get_arg_val<uint32_t>(base + 5);
                const uint32_t mask = get_arg_val<uint32_t>(base + 6);
                const uint32_t need = (entry * num_kinds + (kind == 3 ? 1 : 0)) * S_in + rank + 1;
                for (uint32_t pi = 0; pi < num_ports; ++pi) {
                    if (!((mask >> pi) & 1)) {
                        continue;
                    }
                    while (seen[pi] < need) {
                        noc_async_read(get_noc_addr(port_x[pi], port_y[pi], arrival_addr), poll_addr, 4);
                        noc_async_read_barrier();
                        seen[pi] = *scratch;
                    }
                }
            }
            for (uint32_t t0 = 0; t0 < valid; t0 += Sk, ++kb) {
                const uint32_t skv = (valid - t0) < Sk ? (valid - t0) : Sk;
                const bool zs = (i == 0) && (kb >= ZONE_SAMPLE_KB0) && (kb < ZONE_SAMPLE_KB0 + ZONE_SAMPLE_NKB);
                // ---- load_kv_block ----
                {
                    MaybeDeviceZoneScopeIf("reader_kv_reserve", zs);  // back-pressure: compute not done with block
                    cb_reserve_back(cb_k, Sk * DHt);
                    cb_reserve_back(cb_v, Sk * DVt);
                }
                const uint32_t kw = get_write_ptr(cb_k);
                const uint32_t vw = get_write_ptr(cb_v);
                if (mc_recv) {
                    MaybeDeviceZoneScopeIf("reader_kv_mc_recv", zs);
                    kv_receiver->receive();  // K block (slot reserved above: ack = slot free)
                    kv_receiver->receive();  // V block
                    cb_push_back(cb_k, Sk * DHt);
                    cb_push_back(cb_v, Sk * DVt);
                    continue;
                }
                {
                    MaybeDeviceZoneScopeIf("reader_kv_issue", zs);
                    for (uint32_t t = 0; t < skv; ++t) {
                        const uint32_t kpage = k_base + (t0 + t) * DHt;
                        const uint32_t vpage = v_base + (t0 + t) * DVt;
                        for (uint32_t d = 0; d < DHt; ++d) {
                            const uint32_t dst = kw + (d * Sk + t) * kv_tile;
                            if (kind == 0) {
                                noc_async_read_tile(kpage + d, k_acc, dst);
                            } else if (kind == 2) {
                                noc_async_read_tile(kpage + d, lk_acc, dst);
                            } else if (kind == 3) {
                                noc_async_read_tile(kpage + d, ljk_acc, dst);
                            } else {
                                if constexpr (has_joint) {
                                    const auto jk_acc = TensorAccessor(jk_args, jk_addr, kv_tile);
                                    noc_async_read_tile(kpage + d, jk_acc, dst);
                                }
                            }
                        }
                        for (uint32_t d = 0; d < DVt; ++d) {
                            const uint32_t dst = vw + (t * DVt + d) * kv_tile;
                            if (kind == 0) {
                                noc_async_read_tile(vpage + d, v_acc, dst);
                            } else if (kind == 2) {
                                noc_async_read_tile(vpage + d, lv_acc, dst);
                            } else if (kind == 3) {
                                noc_async_read_tile(vpage + d, ljv_acc, dst);
                            } else {
                                if constexpr (has_joint) {
                                    const auto jv_acc = TensorAccessor(jv_args, jv_addr, kv_tile);
                                    noc_async_read_tile(vpage + d, jv_acc, dst);
                                }
                            }
                        }
                    }
                }
                {
                    MaybeDeviceZoneScopeIf("reader_kv_barrier", zs);
                    noc_async_read_barrier();
                }
                if (mc_send) {
                    MaybeDeviceZoneScopeIf("reader_kv_mc_send", zs);
                    kv_sender->send(kw, kw, k_block_bytes);  // waits the members' acks, multicasts, signals
                    kv_sender->send(vw, vw, v_block_bytes);
                }
                cb_push_back(cb_k, Sk * DHt);
                cb_push_back(cb_v, Sk * DVt);
            }
        }
    }

    // ---- release_ports: this core polls no port counter any more ----
    for (uint32_t pi = 0; pi < num_ports; ++pi) {
        noc_semaphore_inc(get_noc_addr(port_x[pi], port_y[pi], cdone_addr), 1);
    }
    noc_async_atomic_barrier();
}
