// SPDX-License-Identifier: Apache-2.0
//
// Reader for chunked-parallel DeltaNet prefill. One core per v-head h. Pure tile
// loader: streams per-chunk post-conv / pre-scaled inputs from DRAM, seeds state once,
// and generates the constant masks (identity, tril) + RMS scaler/eps tiles.
//
// Layouts: q/k/v/z/Kdec/Qd/dcol/betacol/dlast are [Hv*Sp, D] -> tile-row (h*nC + c).
//          KiT is [Hv*Dk, Sp] -> tile (h*Dk_tiles + dk)*nC + c.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(uint32_t));
    return static_cast<uint16_t>((bits + 0x8000) >> 16);
}

// element (r,c) of a 32x32 TILE-layout tile -> linear uint16 index (4 faces of 16x16)
FORCE_INLINE uint32_t tile_idx(uint32_t r, uint32_t c) {
    uint32_t face = (r / 16) * 2 + (c / 16);
    return face * 256 + (r % 16) * 16 + (c % 16);
}

void kernel_main() {
    constexpr uint32_t cbK=get_compile_time_arg_val(0); constexpr uint32_t cbQ=get_compile_time_arg_val(1);
    constexpr uint32_t cbV=get_compile_time_arg_val(2); constexpr uint32_t cbZ=get_compile_time_arg_val(3);
    constexpr uint32_t cbKdec=get_compile_time_arg_val(4); constexpr uint32_t cbKiT=get_compile_time_arg_val(5);
    constexpr uint32_t cbQd=get_compile_time_arg_val(6); constexpr uint32_t cbDcol=get_compile_time_arg_val(7);
    constexpr uint32_t cbBetacol=get_compile_time_arg_val(8); constexpr uint32_t cbDlast=get_compile_time_arg_val(9);
    constexpr uint32_t cbNormW=get_compile_time_arg_val(10); constexpr uint32_t cbScaler=get_compile_time_arg_val(11);
    constexpr uint32_t cbEps=get_compile_time_arg_val(12); constexpr uint32_t cbState=get_compile_time_arg_val(13);
    constexpr uint32_t cbIdent=get_compile_time_arg_val(14); constexpr uint32_t cbTrils=get_compile_time_arg_val(15);
    constexpr uint32_t cbTrili=get_compile_time_arg_val(16);
    constexpr uint32_t Dk_tiles=get_compile_time_arg_val(17); constexpr uint32_t Dv_tiles=get_compile_time_arg_val(18);
    constexpr uint32_t nC=get_compile_time_arg_val(19); constexpr uint32_t state_tiles=get_compile_time_arg_val(20);
    constexpr auto acc_args = TensorAccessorArgs<21>();

    const uint32_t k_addr=get_arg_val<uint32_t>(0); const uint32_t q_addr=get_arg_val<uint32_t>(1);
    const uint32_t v_addr=get_arg_val<uint32_t>(2); const uint32_t z_addr=get_arg_val<uint32_t>(3);
    const uint32_t kdec_addr=get_arg_val<uint32_t>(4); const uint32_t kit_addr=get_arg_val<uint32_t>(5);
    const uint32_t qd_addr=get_arg_val<uint32_t>(6); const uint32_t dcol_addr=get_arg_val<uint32_t>(7);
    const uint32_t beta_addr=get_arg_val<uint32_t>(8); const uint32_t dlast_addr=get_arg_val<uint32_t>(9);
    const uint32_t state_addr=get_arg_val<uint32_t>(10); const uint32_t normw_addr=get_arg_val<uint32_t>(11);
    const uint32_t head_idx=get_arg_val<uint32_t>(12); const uint32_t row_base=get_arg_val<uint32_t>(13);
    const uint32_t kit_row_base=get_arg_val<uint32_t>(14); const uint32_t state_base=get_arg_val<uint32_t>(15);

    const uint32_t tb = get_tile_size(cbK);
    const auto k_acc=TensorAccessor(acc_args,k_addr,tb); const auto q_acc=TensorAccessor(acc_args,q_addr,tb);
    const auto v_acc=TensorAccessor(acc_args,v_addr,tb); const auto z_acc=TensorAccessor(acc_args,z_addr,tb);
    const auto kdec_acc=TensorAccessor(acc_args,kdec_addr,tb); const auto kit_acc=TensorAccessor(acc_args,kit_addr,tb);
    const auto qd_acc=TensorAccessor(acc_args,qd_addr,tb); const auto dcol_acc=TensorAccessor(acc_args,dcol_addr,tb);
    const auto beta_acc=TensorAccessor(acc_args,beta_addr,tb); const auto dlast_acc=TensorAccessor(acc_args,dlast_addr,tb);
    const auto state_acc=TensorAccessor(acc_args,state_addr,tb); const auto normw_acc=TensorAccessor(acc_args,normw_addr,tb);

    // helper: read `n` consecutive tiles starting at `base` into `cb`
    auto read_block = [&](uint32_t cb, const auto& acc, uint32_t base, uint32_t n) {
        cb_reserve_back(cb, n);
        uint32_t l1 = get_write_ptr(cb);
        for (uint32_t t = 0; t < n; t++) { noc_async_read_tile(base + t, acc, l1); l1 += tb; }
        noc_async_read_barrier();
        cb_push_back(cb, n);
    };
    auto gen_mask = [&](uint32_t cb, uint32_t mode) {  // 0=ident,1=strict-lower,2=incl-lower
        cb_reserve_back(cb, 1);
        volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb));
        for (uint32_t i = 0; i < 1024; i++) t[i] = 0;
        uint16_t one = f32_to_bf16(1.0f);
        for (uint32_t r = 0; r < 32; r++)
            for (uint32_t c = 0; c < 32; c++) {
                bool set = (mode == 0) ? (r == c) : (mode == 1) ? (r > c) : (r >= c);
                if (set) t[tile_idx(r, c)] = one;
            }
        cb_push_back(cb, 1);
    };
    auto gen_fill = [&](uint32_t cb, float val) {  // fill whole tile with val
        cb_reserve_back(cb, 1);
        volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb));
        uint16_t v = f32_to_bf16(val);
        for (uint32_t i = 0; i < 1024; i++) t[i] = v;
        cb_push_back(cb, 1);
    };

    // once
    read_block(cbState, state_acc, state_base, state_tiles);
    gen_mask(cbIdent, 0); gen_mask(cbTrils, 1); gen_mask(cbTrili, 2);

    const float inv_dv = 1.0f / static_cast<float>(Dv_tiles * 32);
    for (uint32_t c = 0; c < nC; c++) {
        uint32_t rk = (row_base + c) * Dk_tiles;
        uint32_t rv = (row_base + c) * Dv_tiles;
        read_block(cbK, k_acc, rk, Dk_tiles);
        read_block(cbQ, q_acc, rk, Dk_tiles);
        read_block(cbKdec, kdec_acc, rk, Dk_tiles);
        read_block(cbQd, qd_acc, rk, Dk_tiles);
        read_block(cbV, v_acc, rv, Dv_tiles);
        read_block(cbZ, z_acc, rv, Dv_tiles);
        read_block(cbDcol, dcol_acc, rv, Dv_tiles);
        read_block(cbBetacol, beta_acc, rv, Dv_tiles);
        read_block(cbNormW, normw_acc, 0, Dv_tiles);
        read_block(cbDlast, dlast_acc, rv, 1);
        // KiT: Dk_tiles row-tiles, col-tile c
        cb_reserve_back(cbKiT, Dk_tiles);
        uint32_t l1 = get_write_ptr(cbKiT);
        for (uint32_t dk = 0; dk < Dk_tiles; dk++) { noc_async_read_tile(kit_row_base + dk * nC + c, kit_acc, l1); l1 += tb; }
        noc_async_read_barrier();
        cb_push_back(cbKiT, Dk_tiles);
        // RMS scaler (1/Dv broadcast) + eps (filled)
        gen_fill(cbScaler, inv_dv);
        gen_fill(cbEps, 1e-6f);
    }
}
