// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for gated_delta_attn_preprocess. Drains the eight compute-produced output CBs and writes
// them to DRAM in the exact layout consumed by gated_delta_attn_seq.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);

    const uint32_t core_work_start = get_arg_val<uint32_t>(0);
    const uint32_t num_work = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_arg_val<uint32_t>(2);
    const uint32_t lu_addr = get_arg_val<uint32_t>(3);
    const uint32_t vbs_addr = get_arg_val<uint32_t>(4);
    const uint32_t kbs_addr = get_arg_val<uint32_t>(5);
    const uint32_t att_addr = get_arg_val<uint32_t>(6);
    const uint32_t qdec_addr = get_arg_val<uint32_t>(7);
    const uint32_t kdt_addr = get_arg_val<uint32_t>(8);
    const uint32_t dle_addr = get_arg_val<uint32_t>(9);
    const uint32_t linv_addr = get_arg_val<uint32_t>(10);
    const uint32_t work_stride = get_arg_val<uint32_t>(11);

    constexpr uint32_t attn_tiles = Ct * Ct;
    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t in_kv_tiles = Ct * Kt;
    constexpr uint32_t kdt_tiles = Kt * Ct;
    constexpr uint32_t linv_tiles = Ct;
    constexpr uint32_t cb_lu = tt::CBIndex::c_10;
    constexpr uint32_t cb_vbs = tt::CBIndex::c_11;
    constexpr uint32_t cb_kbs = tt::CBIndex::c_12;
    constexpr uint32_t cb_att = tt::CBIndex::c_13;
    constexpr uint32_t cb_qdec = tt::CBIndex::c_14;
    constexpr uint32_t cb_kdt = tt::CBIndex::c_15;
    constexpr uint32_t cb_dle = tt::CBIndex::c_16;
    constexpr uint32_t cb_linv = tt::CBIndex::c_17;
    constexpr uint32_t f32_tile = get_tile_size(cb_lu);

    constexpr auto lu_args = TensorAccessorArgs<3>();
    const auto lu_gen = TensorAccessor(lu_args, lu_addr, f32_tile);
    constexpr auto vbs_args = TensorAccessorArgs<lu_args.next_compile_time_args_offset()>();
    const auto vbs_gen = TensorAccessor(vbs_args, vbs_addr, f32_tile);
    constexpr auto kbs_args = TensorAccessorArgs<vbs_args.next_compile_time_args_offset()>();
    const auto kbs_gen = TensorAccessor(kbs_args, kbs_addr, f32_tile);
    constexpr auto att_args = TensorAccessorArgs<kbs_args.next_compile_time_args_offset()>();
    const auto att_gen = TensorAccessor(att_args, att_addr, f32_tile);
    constexpr auto qdec_args = TensorAccessorArgs<att_args.next_compile_time_args_offset()>();
    const auto qdec_gen = TensorAccessor(qdec_args, qdec_addr, f32_tile);
    constexpr auto kdt_args = TensorAccessorArgs<qdec_args.next_compile_time_args_offset()>();
    const auto kdt_gen = TensorAccessor(kdt_args, kdt_addr, f32_tile);
    constexpr auto dle_args = TensorAccessorArgs<kdt_args.next_compile_time_args_offset()>();
    const auto dle_gen = TensorAccessor(dle_args, dle_addr, f32_tile);
    constexpr auto linv_args = TensorAccessorArgs<dle_args.next_compile_time_args_offset()>();
    const auto linv_gen = TensorAccessor(linv_args, linv_addr, f32_tile);

    Noc noc;
    CircularBuffer lu(cb_lu);
    CircularBuffer vbs(cb_vbs);
    CircularBuffer kbs(cb_kbs);
    CircularBuffer att(cb_att);
    CircularBuffer qdec(cb_qdec);
    CircularBuffer kdt(cb_kdt);
    CircularBuffer dle(cb_dle);
    CircularBuffer linv(cb_linv);

    for (uint32_t work = core_work_start; work < num_work; work += work_stride) {
        const uint32_t h = work / NC;
        const uint32_t c = work - h * NC;
        const uint32_t off_att = h * NC * attn_tiles + c * attn_tiles;
        const uint32_t off_out = h * NC * out_tiles + c * out_tiles;
        const uint32_t off_kv = h * NC * in_kv_tiles + c * in_kv_tiles;
        const uint32_t off_kdt = h * NC * kdt_tiles + c * kdt_tiles;
        const uint32_t off_dle = h * NC + c;
        const uint32_t off_linv = h * NC * linv_tiles + c * linv_tiles;

        lu.wait_front(attn_tiles);
        for (uint32_t t = 0; t < attn_tiles; t++) {
            noc.async_write(lu, lu_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = off_att + t});
        }
        noc.async_write_barrier();
        lu.pop_front(attn_tiles);

        att.wait_front(attn_tiles);
        for (uint32_t t = 0; t < attn_tiles; t++) {
            noc.async_write(att, att_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = off_att + t});
        }
        noc.async_write_barrier();
        att.pop_front(attn_tiles);

        vbs.wait_front(out_tiles);
        for (uint32_t t = 0; t < out_tiles; t++) {
            noc.async_write(vbs, vbs_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = off_out + t});
        }
        noc.async_write_barrier();
        vbs.pop_front(out_tiles);

        kbs.wait_front(in_kv_tiles);
        for (uint32_t t = 0; t < in_kv_tiles; t++) {
            noc.async_write(kbs, kbs_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = off_kv + t});
        }
        noc.async_write_barrier();
        kbs.pop_front(in_kv_tiles);

        qdec.wait_front(in_kv_tiles);
        for (uint32_t t = 0; t < in_kv_tiles; t++) {
            noc.async_write(qdec, qdec_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = off_kv + t});
        }
        noc.async_write_barrier();
        qdec.pop_front(in_kv_tiles);

        kdt.wait_front(kdt_tiles);
        for (uint32_t t = 0; t < kdt_tiles; t++) {
            noc.async_write(kdt, kdt_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = off_kdt + t});
        }
        noc.async_write_barrier();
        kdt.pop_front(kdt_tiles);

        dle.wait_front(1);
        noc.async_write(dle, dle_gen, f32_tile, {.offset_bytes = 0}, {.page_id = off_dle});
        noc.async_write_barrier();
        dle.pop_front(1);

        linv.wait_front(linv_tiles);
        for (uint32_t t = 0; t < linv_tiles; t++) {
            noc.async_write(linv, linv_gen, f32_tile, {.offset_bytes = t * f32_tile}, {.page_id = off_linv + t});
        }
        noc.async_write_barrier();
        linv.pop_front(linv_tiles);
    }
}
