// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader: constants (eye, tril) once, initial state S once (from s0, a required input),
// then per-chunk q,k,v,g,beta; g and beta are token-major [B, T, HV] (chunk_gdn_gate_read.hpp). Device 2.0 API.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "chunk_gdn_gate_read.hpp"

// CB indices (must match program factory).
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7, cb_S = 8;

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
    constexpr auto s0_a = TensorAccessorArgs<ones_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t NC = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);
    const uint32_t b_addr = get_arg_val<uint32_t>(6);
    const uint32_t eye_addr = get_arg_val<uint32_t>(7);
    const uint32_t tril_addr = get_arg_val<uint32_t>(8);
    const uint32_t ones_addr = get_arg_val<uint32_t>(9);
    const uint32_t s0_addr = get_arg_val<uint32_t>(10);
    // The gates' source: value heads per batch row, tokens T (rows past it are zero) and each tensor's tile size.
    const uint32_t HV = get_arg_val<uint32_t>(11);
    const uint32_t T = get_arg_val<uint32_t>(12);
    const uint32_t g_tile_bytes = get_arg_val<uint32_t>(13);
    const uint32_t b_tile_bytes = get_arg_val<uint32_t>(14);

    // Mixed precision: q/k/v are bf16 (half tile size); constants/state and the gate CBs are fp32 (the gates'
    // sources are fp32 or bf16). The DRAM page size per accessor MUST match each tensor's own tile size, else
    // fp32 tensors read at the bf16 stride (and vice versa) and produce garbage.
    const uint32_t tb_io = get_tile_size(cb_q);  // bf16 q/k/v
    const uint32_t tb_f = get_tile_size(cb_g);   // fp32 g/beta/eye/tril/ones/state
    const auto q_acc = TensorAccessor(q_a, q_addr, tb_io);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb_io);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb_io);
    const auto g_acc = TensorAccessor(g_a, g_addr, g_tile_bytes);
    const auto b_acc = TensorAccessor(b_a, b_addr, b_tile_bytes);
    const gdn_gate::Source g_src = gdn_gate::source(T, HV, g_tile_bytes);
    const gdn_gate::Source b_src = gdn_gate::source(T, HV, b_tile_bytes);
    const auto eye_acc = TensorAccessor(eye_a, eye_addr, tb_f);
    const auto tril_acc = TensorAccessor(tril_a, tril_addr, tb_f);
    const auto ones_acc = TensorAccessor(ones_a, ones_addr, tb_f);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb_f);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;

    Noc noc;

    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n, uint32_t tb) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // constants (once)
    read_into(eye_acc, cb_eye, 0, cc, tb_f);
    read_into(tril_acc, cb_tril, 0, cc, tb_f);
    read_into(ones_acc, cb_ones, 0, cc, tb_f);

    // initial state S (once) — host always provides it (zeros if none).
    read_into(s0_acc, cb_S, h * kv, kv, tb_f);

    for (uint32_t c = 0; c < NC; c++) {
        const uint32_t hc = h * NC + c;
        read_into(q_acc, cb_q, hc * ck, ck, tb_io);
        read_into(k_acc, cb_k, hc * ck, ck, tb_io);
        read_into(v_acc, cb_v, hc * cv, cv, tb_io);
        // The gates: the token-major tiles holding head h's column, settled into column 0 once landed.
        gdn_gate::issue(noc, g_acc, cb_g, h, c, Ct, g_src, tb_f);
        gdn_gate::issue(noc, b_acc, cb_beta, h, c, Ct, b_src, tb_f);
        noc.async_read_barrier();
        gdn_gate::settle(cb_g, h, c, Ct, g_src, tb_f);
        gdn_gate::settle(cb_beta, h, c, Ct, b_src, tb_f);
        CircularBuffer(cb_g).push_back(Ct);
        CircularBuffer(cb_beta).push_back(Ct);
    }
}
