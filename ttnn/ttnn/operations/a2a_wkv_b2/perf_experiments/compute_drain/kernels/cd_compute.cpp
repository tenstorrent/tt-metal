// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// compute_drain bench compute, one head, Mt tile rows. Variant selected by CD_VARIANT:
//   0 BASE      : the op's compute verbatim — per row tilize<Kt> -> cb_attn_tiled, then matmul_block
//                 [1 x Kt] @ [Kt x Nt] (K in DEST, one block) -> cb_out.
//   1 LOOKAHEAD : software-pipelined order tilize(r+1) before matmul(r) (cb_attn_tiled holds 2 rows), so
//                 the matmul's in0 is already packed when it starts. The last row keeps the base order.
//   2 KSPLIT    : row in KC K-chunks (KC_OTHER / KC_LAST, the reader lands chunk by chunk): one matmul_block
//                 call with num_k_blocks = KC whose PreKBlockFn tilizes chunk c (InitMode::ShortAfterPreKBlock
//                 restores matmul state); partials spill to cb_interm (bf16 = the 16-bit DEST format, so the
//                 spill/reload is lossless) and reload. W stays resident (in1 NoWaitNoPop + per-block offset).
// The precision contract (HiFi2, fp32_dest_acc_en=False, bf16 in, bf8_b W/out) is identical for all variants.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "api/debug/kernel_profiler.hpp"

using namespace compute_kernel_lib;

constexpr uint32_t cb_attn_rm = get_compile_time_arg_val(0);
constexpr uint32_t cb_attn_tiled = get_compile_time_arg_val(1);
constexpr uint32_t cb_weight = get_compile_time_arg_val(2);
constexpr uint32_t cb_out = get_compile_time_arg_val(3);
constexpr uint32_t Kt = get_compile_time_arg_val(4);
constexpr uint32_t Nt = get_compile_time_arg_val(5);
constexpr uint32_t Mt = get_compile_time_arg_val(6);
constexpr uint32_t cb_interm = get_compile_time_arg_val(7);
constexpr uint32_t KC_OTHER = get_compile_time_arg_val(8);
constexpr uint32_t KC_LAST = get_compile_time_arg_val(9);

ALWI void base_tilize() { tilize<Kt, cb_attn_rm, cb_attn_tiled>(1); }

ALWI void base_matmul(CircularBuffer& in0, CircularBuffer& in1, CircularBuffer& out) {
    matmul_block<
        false,
        false,
        LastBlockTarget::Out,
        OutputCBLayout::SubblockMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::WaitAndRetainOnLastBlock>(in0, in1, out, out, MatmulBlockShape::of(1, 1, 1, Nt, Kt, 1));
}

template <uint32_t kc>
struct TilizeChunk {
    ALWI void operator()(uint32_t /*block*/, uint32_t /*nkb*/, bool last) const {
        tilize<kc, cb_attn_rm, cb_attn_tiled>(1);
        // the helper (reconfig = INPUT) leaves the packer to us: spills in interm's bf16, last block in out's bf8_b
        PACK((pack_reconfig_data_format(last ? cb_out : cb_interm)));
    }
};

template <uint32_t kc>
struct In1Off {
    ALWI uint32_t operator()(uint32_t block) const { return block * kc * Nt; }
};

template <uint32_t KC>
ALWI void ksplit_row(CircularBuffer& in0, CircularBuffer& in1, CircularBuffer& out, CircularBuffer& interm) {
    constexpr uint32_t kc = Kt / KC;
    matmul_block<
        false,
        false,
        LastBlockTarget::Out,
        OutputCBLayout::SubblockMajor,
        matmul_config::InitMode::ShortAfterPreKBlock,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::NoWaitNoPop,
        NoPostCompute,
        TilizeChunk<kc>,
        NoPostKBlock,
        0,
        NoKBlockInnerDimFn,
        NoIn0Source,
        In1Off<kc>,
        false,
        NoneActivation,
        matmul_config::DataFormatReconfig::INPUT>(
        in0,
        in1,
        out,
        interm,
        MatmulBlockShape::of(1, 1, 1, Nt, kc, KC),
        {},
        TilizeChunk<kc>{},
        Nt,
        0,
        {},
        {},
        {},
        In1Off<kc>{});
}

// ---- raw-LLK chunked tilize + K-chunk-waiting matmul (variants 3, 4) ----
// Bypasses compute_kernel_lib::tilize (capability): the helper pushes cb_attn_tiled once per block (the
// whole Kt-wide row), so the matmul can only start once the LAST tilized tile is packed. Here the same BH
// fast-tilize LLK sequence the helper runs (fast_tilize_block body) is issued one 4-tile column chunk at a
// time (unpack column offset c*4 of the full-width RM row, init once per row with full_dim = Kt) and each
// chunk is pushed as soon as it is packed. Bypasses compute_kernel_lib::matmul_block (capability): the helper
// waits for the whole in0 K-block before the first K step; here the K loop waits per tilize chunk, K stays
// in one DEST block (same accumulation order => bit-identical).
constexpr uint32_t TCH = 4;  // fast-tilize unit (tiles per chunk)
ALWI void chunked_tilize_row() {
    reconfig_data_format_srca(cb_attn_rm);
    pack_reconfig_data_format(cb_attn_tiled);
    fast_tilize_init(cb_attn_rm, Kt, cb_attn_tiled);
    cb_wait_front(cb_attn_rm, Kt);
    for (uint32_t c = 0; c < Kt / TCH; ++c) {
        cb_reserve_back(cb_attn_tiled, TCH);
        PACK((llk_pack_fast_tilize_row_begin(cb_attn_tiled, 0)));
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        UNPACK((llk_unpack_fast_tilize_block(cb_attn_rm, 0, TCH, c * TCH)));
        MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_attn_rm, 4)));
        PACK((llk_pack_fast_tilize_row_chunk(0, cb_attn_tiled, TCH)));
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_fast_tilize_row_end()));
        cb_push_back(cb_attn_tiled, TCH);
    }
    cb_pop_front(cb_attn_rm, Kt);
    fast_tilize_uninit(cb_attn_rm, cb_attn_tiled, Kt);
}

ALWI void chunk_wait_matmul_row() {
    reconfig_data_format(cb_weight, cb_attn_tiled);
    pack_reconfig_data_format(cb_out);
    ckernel::matmul_block_init(cb_attn_tiled, cb_weight, false, Nt, 1, Kt);
    tile_regs_acquire();
    for (uint32_t k = 0; k < Kt; ++k) {
        if (k % TCH == 0) {
            cb_wait_front(cb_attn_tiled, k + TCH);
        }
        ckernel::matmul_block(cb_attn_tiled, cb_weight, k, k * Nt, 0, false, Nt, 1, Kt);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, Nt);
    tile_regs_wait();
    pack_block(0, cb_out, Nt);
    tile_regs_release();
    cb_push_back(cb_out, Nt);
    cb_pop_front(cb_attn_tiled, Kt);
}

// ---- variant 5: tilize fused into the matmul's in0 unpack (raw LLK) ----
// Bypasses BOTH compute_kernel_lib::tilize and compute_kernel_lib::matmul_block (capability): neither the
// helpers nor any LLK expose "matmul with a row-major in0". The matmul maps in0 -> SrcB through unpacker 1
// (THCON SEC1), which has the same Tileize_mode / shift_amount config as unpacker 0. SEC1 is put in tileize
// mode with a 1x16 face read (x_dim = 16, x_end = 255, shift_amount = RM row pitch in 16 B words) and every
// in0 tile is unpacked face by face exactly like the BH 8-bit tilize path (_llk_unpack_tilize_, 4 UNPACRs
// with Ch0.Z/Ch1.Z++ and a CFGSHIFTMASK of the base by SCRATCH between the top and bottom face pairs), from
// the RM tile row in cb_attn_rm at column offset k * 64 B. SrcB then holds the tilized tile, so the FPU work
// and DEST accumulation order are unchanged: no tilize pack, no cb_attn_tiled round trip, no per-row re-inits.
// (A whole-tile tileize read — x_dim = 1024 — lands the rows linearly, not in face order: measured PCC 0.06.)
#ifdef TRISC_UNPACK
constexpr uint32_t ROW_WORDS = (Kt * 32 * 2) >> 4;  // RM row pitch in 16 B words
ALWI void srcb_tilize_cfg(bool on) {
    const uint32_t x_dim = on ? FACE_C_DIM : FACE_R_DIM * FACE_C_DIM;
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, TILE_DESC_UPPER_HALFWORD_MASK>(x_dim << 16);
    unpack_config_u config = {0};
    config.f.out_data_format = unpack_dst_format[cb_attn_rm];
    config.f.throttle_mode = 2;
    config.f.tileize_mode = on ? 1 : 0;
    config.f.shift_amount = on ? ROW_WORDS : 0;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG2_Out_data_format_ADDR32);
    if (on) {
        // bottom face pair: base += 16 rows * pitch, minus the Z=2 offset (2 * 16 datums * 2 B = 4 words)
        const uint32_t inc = ROW_WORDS * FACE_R_DIM - 4;
        TT_SETDMAREG(0, LOWER_HALFWORD(inc), 0, LO_16(p_gpr_unpack::TMP0));
        TT_SETDMAREG(0, UPPER_HALFWORD(inc), 0, HI_16(p_gpr_unpack::TMP0));
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32);
        TT_SETADCXX(p_setadc::UNP_B, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    } else {
        TT_SETADCXX(p_setadc::UNP_B, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    }
    TTI_NOP;
}

ALWI void unpack_tilized_in0_step(uint32_t k) {
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    const uint32_t tsz_b = get_local_cb_interface(cb_weight).fifo_page_size;
    const uint32_t addr_a = get_local_cb_interface(cb_attn_rm).fifo_rd_ptr - 1 + 4 * k;
    const uint32_t addr_b = get_local_cb_interface(cb_weight).fifo_rd_ptr - 1 + tsz_b * (k * Nt);
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(addr_b, addr_a, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101);
    TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    if (0 == unp_cfg_context) {
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC1_REG3_Base_address_ADDR32);
    } else {
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
    }
    TTI_NOP;
    TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101);
    TT_MOP(0, Nt - 1, unp_cfg_context == 0 ? 0 : 0xff);  // the matmul MOP: Nt in1 tiles -> SrcA
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

ALWI void fused_matmul_row() {
    cb_wait_front(cb_attn_rm, Kt);
    tile_regs_acquire();
    for (uint32_t k = 0; k < Kt; ++k) {
        UNPACK((unpack_tilized_in0_step(k)));
        MATH((matmul_block_math_dynamic_throttle(cb_attn_rm, cb_weight, 0, false, Nt, 1)));
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, Nt);
    tile_regs_wait();
    pack_block(0, cb_out, Nt);
    tile_regs_release();
    cb_push_back(cb_out, Nt);
    cb_pop_front(cb_attn_rm, Kt);
}

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_attn_tiled, cb_weight, cb_out);
    CircularBuffer in0(cb_attn_tiled), in1(cb_weight), out(cb_out), interm(cb_interm);

#if CD_VARIANT == 0
    for (uint32_t r = 0; r < Mt; ++r) {
        {
            DeviceZoneScopedN("compute_tilize");
            base_tilize();
        }
        {
            DeviceZoneScopedN("compute_matmul");
            base_matmul(in0, in1, out);
        }
    }
#elif CD_VARIANT == 1
    {
        DeviceZoneScopedN("compute_tilize");
        base_tilize();
    }
    for (uint32_t r = 0; r < Mt; ++r) {
        if (r + 1 < Mt - 1) {
            DeviceZoneScopedN("compute_tilize");
            base_tilize();
        }
        {
            DeviceZoneScopedN("compute_matmul");
            base_matmul(in0, in1, out);
        }
        if (r + 1 == Mt - 1) {
            DeviceZoneScopedN("compute_tilize");
            base_tilize();
        }
    }
#elif CD_VARIANT == 2
    cb_wait_front(cb_weight, Kt * Nt);
    for (uint32_t r = 0; r < Mt; ++r) {
        DeviceZoneScopedN("compute_matmul");
        if (r == Mt - 1) {
            ksplit_row<KC_LAST>(in0, in1, out, interm);
        } else {
            ksplit_row<KC_OTHER>(in0, in1, out, interm);
        }
    }
#elif CD_VARIANT == 3
    cb_wait_front(cb_weight, Kt * Nt);
    for (uint32_t r = 0; r < Mt; ++r) {
        {
            DeviceZoneScopedN("compute_tilize");
            chunked_tilize_row();
        }
        {
            DeviceZoneScopedN("compute_matmul");
            chunk_wait_matmul_row();
        }
    }
#elif CD_VARIANT == 4
    cb_wait_front(cb_weight, Kt * Nt);
    {
        DeviceZoneScopedN("compute_tilize");
        chunked_tilize_row();
    }
    for (uint32_t r = 0; r < Mt; ++r) {
        if (r + 1 < Mt - 1) {
            DeviceZoneScopedN("compute_tilize");
            chunked_tilize_row();
        }
        {
            DeviceZoneScopedN("compute_matmul");
            chunk_wait_matmul_row();
        }
        if (r + 1 == Mt - 1) {
            DeviceZoneScopedN("compute_tilize");
            chunked_tilize_row();
        }
    }
#elif CD_VARIANT == 5
    reconfig_data_format(cb_weight, cb_attn_rm);
    pack_reconfig_data_format(cb_out);
    ckernel::matmul_block_init(cb_attn_rm, cb_weight, false, Nt, 1, Kt);
    UNPACK((srcb_tilize_cfg(true)));
    cb_wait_front(cb_weight, Kt * Nt);
    for (uint32_t r = 0; r < Mt; ++r) {
        DeviceZoneScopedN("compute_matmul");
        fused_matmul_row();
    }
    UNPACK((srcb_tilize_cfg(false)));
#endif
    cb_pop_front(cb_weight, Kt * Nt);
}
