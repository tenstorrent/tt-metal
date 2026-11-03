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

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
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
    constexpr uint32_t n_q_t = get_compile_time_arg_val(7);
    constexpr uint32_t Lq_t = get_compile_time_arg_val(8);
    constexpr uint32_t n_kv_t = get_compile_time_arg_val(9);
    constexpr uint32_t BH = get_compile_time_arg_val(10);
    constexpr uint32_t C = get_compile_time_arg_val(11);
    constexpr uint32_t hq = get_compile_time_arg_val(12);
    constexpr uint32_t hk = get_compile_time_arg_val(13);
    constexpr uint32_t num_ports = get_compile_time_arg_val(14);
    constexpr bool has_joint = Lq_t > 0;
    constexpr uint32_t q_per_kv = hq / hk;
    static_assert(num_ports <= MAX_PORTS, "too many port cores");

    constexpr auto q_args = TensorAccessorArgs<15>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto jq_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto jk_args = TensorAccessorArgs<jq_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto jv_args = TensorAccessorArgs<jk_args.next_compile_time_args_offset()>();
    constexpr auto lk_args = TensorAccessorArgs<jv_args.next_compile_time_args_offset()>();
    constexpr auto lv_args = TensorAccessorArgs<lk_args.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(a++);
    const uint32_t k_addr = get_arg_val<uint32_t>(a++);
    const uint32_t v_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t jq_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t jk_addr = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t jv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lk_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t core_idx = get_arg_val<uint32_t>(a++);
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
    const uint32_t seg_idx = a;  // per segment: kind, valid_tiles, masked, slot, S_in, rank, port_mask

    const uint32_t q_tile = get_tile_size(cb_q);
    const uint32_t kv_tile = get_tile_size(cb_k);
    const auto q_acc = TensorAccessor(q_args, q_addr, q_tile);
    const auto k_acc = TensorAccessor(k_args, k_addr, kv_tile);
    const auto v_acc = TensorAccessor(v_args, v_addr, kv_tile);
    const auto lk_acc = TensorAccessor(lk_args, lk_addr, kv_tile);
    const auto lv_acc = TensorAccessor(lv_args, lv_addr, kv_tile);

    // landing slot pages: slot * (b*hk*n_kv_t*D) + entry pages; entry = b*hk + kv head
    constexpr uint32_t B = BH / hq;
    constexpr uint32_t k_slot_pages = B * hk * n_kv_t * DHt;
    constexpr uint32_t v_slot_pages = B * hk * n_kv_t * DVt;
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll_addr);

    constexpr uint32_t U = total_units<Sq, n_q_t, Lq_t, BH, C>();
    const uint32_t my_units = units_of_core(U, C, core_idx);

    for (uint32_t i = 0; i < my_units; ++i) {
        MaybeDeviceZoneScope("reader_unit");
        uint32_t kb = 0;  // key blocks issued for this unit (zone sampling)
        const AttnUnit unit = decode_unit<Sq, n_q_t, Lq_t>(core_idx + i * C);
        const uint32_t b = unit.bh / hq;
        const uint32_t h = unit.bh - b * hq;
        const uint32_t entry = b * hk + h / q_per_kv;

        // ---- load_q_block ----
        {
            MaybeDeviceZoneScope("reader_q_reserve");
            cb_reserve_back(cb_q, Sq * DHt);
        }
        {
            MaybeDeviceZoneScope("reader_q_read");
            uint32_t wptr = get_write_ptr(cb_q);
            if (!unit.joint) {
                uint32_t page = (unit.bh * n_q_t + unit.row0) * DHt;
                for (uint32_t t = 0; t < unit.rows * DHt; ++t) {
                    noc_async_read_tile(page + t, q_acc, wptr);
                    wptr += q_tile;
                }
            } else {
                if constexpr (has_joint) {
                    const auto jq_acc = TensorAccessor(jq_args, jq_addr, q_tile);
                    uint32_t page = (unit.bh * Lq_t + unit.row0) * DHt;
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
            const uint32_t base = seg_idx + s * 7;
            const uint32_t kind = get_arg_val<uint32_t>(base + 0);
            const uint32_t valid = get_arg_val<uint32_t>(base + 1);
            const uint32_t slot = get_arg_val<uint32_t>(base + 3);
            uint32_t k_base, v_base;  // first page of this head in the source tensor
            if (kind == 1) {
                k_base = entry * Lq_t * DHt;
                v_base = entry * Lq_t * DVt;
            } else {
                k_base = (kind == 2 ? slot * k_slot_pages : 0) + entry * n_kv_t * DHt;
                v_base = (kind == 2 ? slot * v_slot_pages : 0) + entry * n_kv_t * DVt;
            }
            if (kind == 2) {
                MaybeDeviceZoneScope("reader_landing_wait");
                // ---- await_landing: count needed = entry * S_in + rank + 1 on every port carrying the source ----
                const uint32_t S_in = get_arg_val<uint32_t>(base + 4);
                const uint32_t rank = get_arg_val<uint32_t>(base + 5);
                const uint32_t mask = get_arg_val<uint32_t>(base + 6);
                const uint32_t need = entry * S_in + rank + 1;
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
                {
                    MaybeDeviceZoneScopeIf("reader_kv_issue", zs);
                    const uint32_t kw = get_write_ptr(cb_k);
                    const uint32_t vw = get_write_ptr(cb_v);
                    for (uint32_t t = 0; t < skv; ++t) {
                        const uint32_t kpage = k_base + (t0 + t) * DHt;
                        const uint32_t vpage = v_base + (t0 + t) * DVt;
                        for (uint32_t d = 0; d < DHt; ++d) {
                            const uint32_t dst = kw + (d * Sk + t) * kv_tile;
                            if (kind == 0) {
                                noc_async_read_tile(kpage + d, k_acc, dst);
                            } else if (kind == 2) {
                                noc_async_read_tile(kpage + d, lk_acc, dst);
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
