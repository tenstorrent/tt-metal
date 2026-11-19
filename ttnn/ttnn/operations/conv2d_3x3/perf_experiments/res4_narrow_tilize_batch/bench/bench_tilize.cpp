// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated 1-tile-wide tilize bench (compute only, one core, Blackhole, fp32 DEST half-sync): conv2d_3x3's
// padded_tilize tw == 1 loop (res4.x BLOCK prelude), bf16 row-major slab -> bf8_b tiles. Every tile's source is a
// 32-pixel run of a resident bf16 slab (64 B per pixel) at an address the UNPACK thread computes, as padded_tilize
// does; no data movement kernels. `reps` calls, each one full tilize call (init .. uninit) of `n` tiles, then the
// UNPACK thread drains the output CB (the matmul stand-in).
//
// RAW LLK (bench): the per-tile unpack tilize at a computed address (llk_unpack_tilize_impl), as padded_tilize; the
// tilize helpers read at the CB read pointer.
//
// variant 0: padded_tilize tw == 1, exactly as the op (baseline): one DEST section + CB reserve/push per tile
// variant 1: `batch` tiles per DEST section, per-tile llk_pack at offsets, one CB reserve/push per section
// variant 2: as 1, sections released without ZEROACC (every DEST tile is fully overwritten by the datacopy); the
//            call's last two sections (one per DEST half) still clear, so later DEST accumulators see zeros
// variant 3: as 1, with the CB reserve hoisted to the call (one reserve of n, push per section)
// variant 4: as 1, with a lean pack stream for the section: the tilize pack MOP (the BH 1-wide tilize's half-rate
//            unswizzle PACRs, same PACRs) reprogrammed as one tile per run whose end ops are the hoisted address
//            update of conv2d_3x3_lean_tilize.hpp (STALLWAIT; WRCFG; ADDDMAREG) and a W step, so the section programs
//            its first output address once and runs the MOP per tile (no per-tile llk_pack setup)
// variant 30: fast-tilize window: tile pair p (taps kw = 0 and 2 of one kernel row, d = 1) is ONE BH fast-tilize chunk
//             of 3 tiles read at a 1-pixel row stride (init full_dim 1: tile k of the chunk = pixels src + k .. + 31),
//             one DEST section per pair; the lean pack stream (conv2d_3x3_lean_tilize.hpp) packs chunk tiles 0 and
//             2 and skips 1 (one W step). Needs 2 pixels readable past the kw = 2 run.
// variant 31: BH fast tilize at unit_dim 1 (init full_dim 1, chunk 1), one DEST section per tile, lean pack
// variant 10: diag (wrong output): as 1 without the PACRs (unpack + math + handshakes)
// variant 11: diag (wrong output): as 1 without unpack / datacopy (pack + handshakes only)

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifdef TRISC_UNPACK
template <uint32_t cb_in>
inline void direct_unpack_tilize(uint32_t src) {
    const uint32_t id = get_operand_id(cb_in);
    llk_unpack_tilize_impl(
        (src >> 4) - 1,
        0,
        unpack_src_format[id],
        unpack_dst_format[id],
        get_operand_face_r_dim(id),
        get_operand_num_faces(id),
        get_operand_narrow_tile(id));
}
#endif

#ifdef TRISC_UNPACK
// fast-tilize chunk read from `src` (the op's window_unpack_chunk at col_start 0).
template <uint32_t cb_in>
inline void window_unpack_chunk(uint32_t src, uint32_t chunk) {
    const uint32_t operand_id = get_operand_id(cb_in);
    _llk_unpack_fast_tilize_block_(
        (src >> 4) - 1, 0, unpack_src_format[operand_id], chunk, get_operand_num_faces(operand_id));
}
#endif

#ifdef TRISC_PACK
// conv2d_3x3_lean_tilize.hpp's lean pack stream (copied): 1-tile MOP, hoisted address update.
inline void lean_pack_init() {
    ckernel::load_replay_buf(REPLAY_ADDR_UPDATE_OFFSET, 3, [] {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
    });
    ckernel::ckernel_template tmp(
        1,
        1,
        lltt::replay_insn(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN),
        TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_last_inner_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_last_outer_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_end_ops(lltt::replay_insn(REPLAY_ADDR_UPDATE_OFFSET, 3), TT_OP_NOP);
    tmp.program();
}
inline void lean_row_begin(uint32_t cb_out, uint32_t tile) {
    llk_pack_fast_tilize_row_begin(cb_out, tile);
    TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
}
#endif

#ifdef TRISC_PACK
// Variant 4: the BH tilize pack MOP (_llk_pack_mop_config_<Tilize>, face_r_dim 16: replay [0, 15) + 1 PACR per face
// pair) as inner loop 2, its closing PACR clearing Y and Z (ADDR_MOD_3), then the hoisted address update and W += 1.
inline void lean_tilize_mop_init(uint32_t cb_out) {
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 0, .clr = 1}}.set(ADDR_MOD_3);
    const uint32_t out_id = get_output_id(cb_out);
    const uint32_t tile_l1_size = GET_L1_HEADERLESS_TILE_SIZE(pack_dst_format[out_id]);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_l1_size), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_l1_size), 0, HI_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    ckernel::load_replay_buf(ckernel::packer::replay_buf_offset, 3, [] {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
    });
    constexpr uint32_t sel1 = 0b1010;
    const uint32_t half_end = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_2,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        sel1,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        0);
    const uint32_t tile_end = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_3,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        sel1,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);
    ckernel::ckernel_template tmp(1, 2, lltt::replay_insn(0, 15), half_end);
    tmp.set_last_inner_loop_instr(tile_end);
    tmp.set_last_outer_loop_instr(tile_end);
    tmp.set_end_ops(
        lltt::replay_insn(ckernel::packer::replay_buf_offset, 3), TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.program();
}

// One section: output tiles [0, g) of the reserved block from DEST tiles [0, g).
inline void lean_tilize_pack_section(uint32_t cb_out, uint32_t g) {
    const uint32_t out_id = get_output_id(cb_out);
    const uint32_t addr = get_output_tile_address<true, PackMode::Default>(out_id, 0);
    program_packer_destination(addr);
    TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);  // ch0 Z = W = 0 (and their CRs)
    for (uint32_t i = 0; i < g; ++i) {
        ckernel::ckernel_template::run();
    }
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
}
#endif

template <bool clear>
ALWI void pack_release() {
    PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, clear>()));
}

void kernel_main() {
    constexpr uint32_t CB_SLAB = 0;
    constexpr uint32_t CB_OUT = 16;
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr uint32_t reps = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t span_px = get_compile_time_arg_val(4);  // source runs start in [0, span_px)
    constexpr uint32_t stride_px = get_compile_time_arg_val(5);
    static_assert(batch >= 1 && batch <= 4, "fp32 half-sync DEST holds 4 tiles");

    compute_kernel_hw_startup(CB_SLAB, CB_OUT);
    DataflowBuffer out(CB_OUT);

    [[maybe_unused]] uint32_t slab_l1 = 0;
    UNPACK((slab_l1 = get_local_cb_interface(get_operand_id(CB_SLAB)).fifo_rd_ptr << 4));

    for (uint32_t rep = 0; rep < reps; ++rep) {
        {
            MaybeDeviceZoneScope("bench_tilize");
            reconfig_data_format_srca(CB_SLAB);
            pack_reconfig_data_format(CB_OUT);
            if constexpr (variant != 30 && variant != 31) {
                fast_tilize_init(CB_SLAB, 1, CB_OUT);
            }
            if constexpr (variant == 4) {
                PACK((lean_tilize_mop_init(CB_OUT)));
            }
            if constexpr (variant == 30 || variant == 31) {
                constexpr uint32_t chunk = variant == 30 ? 3 : 1;
                // fast_tilize_init(CB_SLAB, 1, CB_OUT) falls back to the 1-wide tilize: the BH fast-tilize init at
                // full_dim 1 (row stride one pixel), first chunk `chunk`.
                state_configure<Operand::SRCA, Operand::PACK>(CB_SLAB, CB_OUT, __builtin_LINE());
                UNPACK((llk_unpack_fast_tilize_init(CB_SLAB, 1, chunk)));
                MATH((llk_math_fast_tilize_init<DST_ACCUM_MODE>(CB_SLAB)));
                PACK((llk_pack_fast_tilize_init<DST_ACCUM_MODE>(CB_SLAB, CB_OUT, chunk)));
                PACK((lean_pack_init()));
                constexpr uint32_t per = variant == 30 ? 2 : 1;  // output tiles per chunk
                for (uint32_t k = 0; k < n; k += per) {
                    [[maybe_unused]] uint32_t src = 0;
                    UNPACK((src = slab_l1 + (((k / 2) * stride_px) % span_px + 2 * (k % 2)) * 64));
                    out.reserve_back(per);
                    PACK((lean_row_begin(CB_OUT, 0)));
                    MATH((llk_math_wait_for_dest_available()));
                    PACK((llk_packer_wait_for_math_done()));
                    UNPACK((window_unpack_chunk<CB_SLAB>(src, chunk)));
                    MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, CB_SLAB, 4)));
#ifdef TRISC_PACK
                    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
                    ckernel::ckernel_template::run();  // chunk tile 0
                    if constexpr (variant == 30) {
                        TTI_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010);  // skip chunk tile 1 (the center tap)
                        ckernel::ckernel_template::run();                 // chunk tile 2
                    }
#endif
                    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                    out.push_back(per);
                }
                UNPACK((llk_unpack_fast_tilize_uninit<DST_ACCUM_MODE>()));
                MATH((llk_math_fast_tilize_uninit<DST_ACCUM_MODE>(CB_SLAB)));
                PACK((llk_pack_fast_tilize_uninit<DST_ACCUM_MODE>(CB_OUT)));
            } else if constexpr (variant == 0) {
                for (uint32_t k = 0; k < n; ++k) {
                    [[maybe_unused]] uint32_t src = 0;
                    UNPACK((src = slab_l1 + (((k / 2) * stride_px) % span_px + 2 * (k % 2)) * 64));
                    out.reserve_back(1);
                    UNPACK((direct_unpack_tilize<CB_SLAB>(src)));
                    MATH((llk_math_wait_for_dest_available()));
                    PACK((llk_packer_wait_for_math_done()));
                    MATH((llk_math_eltwise_unary_datacopy<
                          DataCopyType::A2D,
                          DST_ACCUM_MODE,
                          BroadcastType::NONE,
                          UnpackToDestEn>(0, CB_SLAB)));
                    PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(0, CB_OUT, 0)));
                    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                    out.push_back(1);
                }
            } else {
                if constexpr (variant == 3) {
                    out.reserve_back(n);
                }
                uint32_t j = 0;  // tiles in the open DEST section
                uint32_t sections_left = (n + batch - 1) / batch;
                for (uint32_t k = 0; k < n; ++k) {
                    if (j == 0) {
                        MATH((llk_math_wait_for_dest_available()));
                        PACK((llk_packer_wait_for_math_done()));
                    }
                    if constexpr (variant != 11) {
                        [[maybe_unused]] uint32_t src = 0;
                        UNPACK((src = slab_l1 + (((k / 2) * stride_px) % span_px + 2 * (k % 2)) * 64));
                        UNPACK((direct_unpack_tilize<CB_SLAB>(src)));
                        MATH((llk_math_eltwise_unary_datacopy<
                              DataCopyType::A2D,
                              DST_ACCUM_MODE,
                              BroadcastType::NONE,
                              UnpackToDestEn>(j, CB_SLAB)));
                    }
                    if (++j == batch || k + 1 == n) {
                        if constexpr (variant != 3) {
                            out.reserve_back(j);
                        }
                        if constexpr (variant == 4) {
                            PACK((lean_tilize_pack_section(CB_OUT, j)));
                        } else if constexpr (variant != 10) {
                            // Offsets are relative to the write pointer, which every push advances.
                            for (uint32_t i = 0; i < j; ++i) {
                                PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(i, CB_OUT, i)));
                            }
                        }
                        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                        if constexpr (variant == 2) {
                            if (sections_left <= 2) {
                                pack_release<true>();
                            } else {
                                pack_release<false>();
                            }
                        } else {
                            PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                        }
                        out.push_back(j);
                        --sections_left;
                        j = 0;
                    }
                }
            }
            if constexpr (variant != 30 && variant != 31) {
                fast_tilize_uninit(CB_SLAB, CB_OUT, 1);
            }
        }
        cb_wait_front(CB_OUT, n);
        cb_pop_front(CB_OUT, n);
    }
}
