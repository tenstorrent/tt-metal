// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bcast_srcb_reuse compute-only bench: one pass of rms_norm's normalize stage over a resident [R x WS]-tile
// block, repeated REPS times (no data movement: every operand is a single-core shard). VARIANT selects the
// pass and its implementation:
//   γ pass   out = x * γ (Row bcast)        1 chain (the op's)   2 stock-unpack only   3 stock-math only
//                                          4 pack only           5 stock per-tile raw  6 HOLD (column-major)
//                                          7 HOLD unpack only    8 HOLD math only
//   inv pass out = x * inv (Col bcast)     11..18 same roles (16 = HOLD row-major)
//   square   sumsq row r = Σ_j x_rj²       21 chain (the op's)   22 stock-unpack only  23 stock-math only
// "X only" variants keep the real work on that thread and replace the other threads with the source-register
// data-valid handshake they would have performed (no DEST sync, no pack): the thread's own floor.

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "bsr_llk.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_x = 1;
constexpr uint32_t cb_sq = 3;
constexpr uint32_t cb_inv = 4;
constexpr uint32_t cb_g = 6;
constexpr uint32_t cb_out = 16;

#ifdef TRISC_UNPACK
// mock unpack: hand math the data-valids it waits for (stall until the bank is free, then set valid)
ALWI void mock_set(uint32_t a, uint32_t b) {
    TT_STALLWAIT(p_stall::STALL_TDMA, (a ? p_stall::SRCA_CLR : 0) | (b ? p_stall::SRCB_CLR : 0));
    TT_SETDVALID((b << 1) | a);
}
#endif
#ifdef TRISC_MATH
// mock math: wait for the valids and clear them
ALWI void mock_clr(uint32_t a, uint32_t b) {
    TT_STALLWAIT(p_stall::STALL_MATH, (a ? p_stall::SRCA_VLD : 0) | (b ? p_stall::SRCB_VLD : 0));
    TT_CLEARDVALID((b << 1) | a, 0);
}
#endif

template <uint32_t V>
constexpr bool is_i = (V >= 11 && V <= 19) || (V >= 31 && V <= 34) || V == 36;

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t WS = get_compile_time_arg_val(1);
    constexpr uint32_t REPS = get_compile_time_arg_val(2);
    constexpr uint32_t V = get_compile_time_arg_val(3);
    constexpr uint32_t SB = get_compile_time_arg_val(4);  // tiles per DEST window
    constexpr uint32_t NT = R * WS;
    constexpr uint32_t cb_b = is_i<V> ? cb_inv : cb_g;  // broadcast operand
    constexpr bool COL = is_i<V>;
    constexpr auto BT = COL ? BroadcastType::COL : BroadcastType::ROW;

    if constexpr (V >= 21 && V < 30) {
        compute_kernel_hw_startup(cb_x, cb_x, cb_sq);
    } else {
        compute_kernel_hw_startup(cb_x, cb_b, cb_out);
    }
    // expose the resident shards
    cb_reserve_back(cb_x, NT);
    cb_push_back(cb_x, NT);
    cb_reserve_back(cb_g, WS);
    cb_push_back(cb_g, WS);
    cb_reserve_back(cb_inv, R);
    cb_push_back(cb_inv, R);
    cb_wait_front(cb_x, NT);
    cb_wait_front(cb_g, WS);
    cb_wait_front(cb_inv, R);
    if constexpr (V < 21 || V >= 30) {
        cb_reserve_back(cb_out, NT);
    }

    constexpr auto x_in = ckl::input(
        cb_x,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto out_strided = ckl::output(
        cb_out,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    constexpr auto inv_in = ckl::input(
        cb_inv, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Col);
    constexpr auto x_block =
        ckl::input(cb_x, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block);

    // one-time inits for the raw variants
    if constexpr (V == 2 || V == 3 || V == 5 || V == 12 || V == 13 || V == 15) {
        MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BT, MATH_FIDELITY>(cb_x, cb_b, 0)));
        UNPACK((llk_unpack_AB_init<BT>(cb_x, cb_b)));
    }
    if constexpr (V == 34 || V == 35) {
        MATH((bsr::math_hold_faces_init<V == 34>()));
        UNPACK((bsr::unpack_hold_faces_init()));
    }
    if constexpr (
        V == 6 || V == 7 || V == 8 || V == 9 || V == 16 || V == 17 || V == 18 || V == 19 || V == 31 || V == 32 ||
        V == 33 || V == 36) {
        MATH((bsr::math_hold_init<COL>()));
        UNPACK((bsr::unpack_hold_init()));
    }
    if constexpr (V == 23) {
        MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY>(
            cb_x, cb_x, 0)));
    }
    if constexpr (V == 22) {
        UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_x, cb_x)));
    }

    for (uint32_t rep = 0; rep < REPS; ++rep) {
        if constexpr (V == 1) {
            ckl::eltwise_chain(
                ckl::IterationShape::grid(R, WS).block_size(SB),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    x_in,
                    ckl::input(
                        cb_g,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Row,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset)>{ckl::StridedTileRange{0, WS}, 0u},
                ckl::PackTile<out_strided>{ckl::StridedTileRange{0, WS}});
        } else if constexpr (V == 11) {
            ckl::eltwise_chain(
                ckl::IterationShape::grid(R, WS).block_size(SB),
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_block, inv_in>{},
                ckl::PackTile<out_strided>{ckl::StridedTileRange{0, WS}});
        } else if constexpr (V == 21) {
            ckl::eltwise_chain(
                ckl::IterationShape::grid(R, WS),
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_in, x_in, ckl::Dst::D0, ckl::DestAccumulation::PerRow>{
                    ckl::StridedTileRange{0, WS}, ckl::StridedTileRange{0, WS}},
                ckl::PackTile<ckl::output(
                    cb_sq,
                    ckl::ReservePolicy::PerOuter,
                    ckl::PushPolicy::PerOuter,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileAddressing::Direct,
                    ckl::DestAccumulation::PerRow)>{});
        } else if constexpr (V == 2 || V == 12 || V == 22) {
            // stock unpack only (row-major), math clears the per-face valids
            for (uint32_t t = 0; t < NT; ++t) {
                const uint32_t r = t / WS, c = t % WS;
                if constexpr (V == 22) {
                    UNPACK((llk_unpack_AB<BroadcastType::NONE>(cb_x, cb_x, t, t)));
                } else {
                    UNPACK((llk_unpack_AB<BT>(cb_x, cb_b, t, COL ? r : c)));
                }
#ifdef TRISC_MATH
                if constexpr (COL) {
                    for (uint32_t fr = 0; fr < 2; ++fr) {
                        mock_clr(1, 0);
                        mock_clr(1, 0);
                        mock_clr(0, 1);
                    }
                } else {
                    for (uint32_t f = 0; f < 4; ++f) {
                        mock_clr(1, 1);
                    }
                }
#endif
            }
        } else if constexpr (V == 3 || V == 13 || V == 23) {
            // stock math only: unpack hands over the per-face valids; no DEST sync
            for (uint32_t t = 0; t < NT; ++t) {
#ifdef TRISC_UNPACK
                if constexpr (COL) {
                    for (uint32_t fr = 0; fr < 2; ++fr) {
                        mock_set(0, 1);
                        mock_set(1, 0);
                        mock_set(1, 0);
                    }
                } else {
                    for (uint32_t f = 0; f < 4; ++f) {
                        mock_set(1, 1);
                    }
                }
#endif
                if constexpr (V == 23) {
                    MATH((llk_math_eltwise_binary<
                          EltwiseBinaryType::ELWMUL,
                          BroadcastType::NONE,
                          DST_ACCUM_MODE,
                          MATH_FIDELITY,
                          EltwiseBinaryReuseDestType::NONE>(cb_x, cb_x, 0, true)));
                } else {
                    MATH((llk_math_eltwise_binary<
                          EltwiseBinaryType::ELWMUL,
                          BT,
                          DST_ACCUM_MODE,
                          MATH_FIDELITY,
                          EltwiseBinaryReuseDestType::NONE>(cb_x, cb_b, t % SB, true)));
                }
            }
        } else if constexpr (V == 4 || V == 14) {
            // pack only (no DEST sync)
            for (uint32_t t = 0; t < NT; ++t) {
                pack_tile<true>(t % SB, cb_out, t);
            }
        } else if constexpr (V == 5 || V == 15) {
            // stock per-tile raw LLK, row-major, SB tiles per DEST window
            for (uint32_t t0 = 0; t0 < NT; t0 += SB) {
                const uint32_t n = (NT - t0) < SB ? NT - t0 : SB;
                tile_regs_acquire();
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t t = t0 + i, r = t / WS, c = t % WS;
                    MATH((llk_math_eltwise_binary<
                          EltwiseBinaryType::ELWMUL,
                          BT,
                          DST_ACCUM_MODE,
                          MATH_FIDELITY,
                          EltwiseBinaryReuseDestType::NONE>(cb_x, cb_b, i, true)));
                    UNPACK((llk_unpack_AB<BT>(cb_x, cb_b, t, COL ? r : c)));
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < n; ++i) {
                    pack_tile<true>(i, cb_out, t0 + i);
                }
                tile_regs_release();
            }
        } else if constexpr (V == 6 || V == 7 || V == 8) {
            // HOLD γ, column-major: per column c, γ_c held in srcB for the R tiles x[0..R-1][c]
            for (uint32_t c = 0; c < WS; ++c) {
#ifdef TRISC_UNPACK
                if constexpr (V == 8) {
                    mock_set(0, 1);
                    for (uint32_t r = 0; r < R; ++r) {
                        mock_set(1, 0);
                    }
                } else {
                    bsr::unpack_hold_run(bsr::unpack_addr(cb_x, c), bsr::unpack_addr(cb_g, c), R, WS);
                }
#endif
                for (uint32_t r0 = 0; r0 < R; r0 += SB) {
                    const uint32_t n = (R - r0) < SB ? R - r0 : SB;
                    if constexpr (V == 6) {
                        tile_regs_acquire();
                    }
#ifdef TRISC_MATH
                    for (uint32_t i = 0; i < n; ++i) {
                        if constexpr (V == 7) {
                            mock_clr(1, 0);
                        } else {
                            bsr::math_hold_tile<false>(i);
                        }
                    }
#endif
                    if constexpr (V == 6) {
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < n; ++i) {
                            pack_tile<true>(i, cb_out, (r0 + i) * WS + c);
                        }
                        tile_regs_release();
                    }
                }
#ifdef TRISC_MATH
                if constexpr (V == 7) {
                    mock_clr(0, 1);
                } else {
                    bsr::math_hold_release_b();
                }
#endif
            }
        } else if constexpr (V == 9) {
            // HOLD γ, column-major, DEST windows of SB tiles spanning columns
            uint32_t slot = 0;
            uint32_t pidx[8];
            for (uint32_t c = 0; c < WS; ++c) {
                UNPACK((bsr::unpack_hold_run(bsr::unpack_addr(cb_x, c), bsr::unpack_addr(cb_g, c), R, WS)));
                for (uint32_t r = 0; r < R; ++r) {
                    if (slot == 0) {
                        tile_regs_acquire();
                    }
                    MATH((bsr::math_hold_tile<false>(slot)));
                    pidx[slot] = r * WS + c;
                    ++slot;
                    if (slot == SB || (c + 1 == WS && r + 1 == R)) {
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < slot; ++i) {
                            pack_tile<true>(i, cb_out, pidx[i]);
                        }
                        tile_regs_release();
                        slot = 0;
                    }
                }
                MATH((bsr::math_hold_release_b()));
            }
        } else if constexpr (V == 31 || V == 32 || V == 33) {
            // diagnostics (inv, row-major): 31 = srcB loaded ONCE for the whole block (timing only, wrong numerics);
            // 32 = srcB in its own unpacker context per row, then the row's srcA tiles in a second context
            uint32_t slot = 0, t0 = 0;
            for (uint32_t r = 0; r < R; ++r) {
#ifdef TRISC_UNPACK
                if constexpr (V == 33) {
                    // one unpacker context per srcA tile; srcB rides in the row's first
                    bsr::unpack_hold_run(bsr::unpack_addr(cb_x, r * WS), bsr::unpack_addr(cb_inv, r), 1, 1);
                    for (uint32_t c = 1; c < WS; ++c) {
                        bsr::unpack_hold_run_pf(bsr::unpack_addr(cb_x, r * WS + c), 0, 1, 1, false);
                    }
                } else {
                    if (V == 32 || r == 0) {
                        bsr::unpack_hold_run_pf(0, bsr::unpack_addr(cb_inv, r), 0, 1, true);
                    }
                    bsr::unpack_hold_run_pf(bsr::unpack_addr(cb_x, r * WS), bsr::unpack_addr(cb_inv, r), WS, 1, false);
                }
#endif
                for (uint32_t c = 0; c < WS; ++c) {
                    if (slot == 0) {
                        tile_regs_acquire();
                    }
                    MATH((bsr::math_hold_tile<true>(slot)));
                    ++slot;
                    if (slot == SB || ((r + 1 == R) && (c + 1 == WS))) {
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < slot; ++i) {
                            pack_tile<true>(i, cb_out, t0 + i);
                        }
                        tile_regs_release();
                        t0 += slot;
                        slot = 0;
                    }
                }
                if (V != 31 || r + 1 == R) {
                    MATH((bsr::math_hold_release_b()));
                }
            }
        } else if constexpr (V == 36) {
            // inv HOLD, fixed-loop windows: each DEST window = RPW whole tile-rows (RPW = SB / WS rows, >= 1)
            constexpr uint32_t RPW = (WS <= SB) ? SB / WS : 1;
            static_assert(WS <= SB, "bench: V36 needs WS <= SB");
            for (uint32_t r0 = 0; r0 < R; r0 += RPW) {
                const uint32_t nr = (R - r0) < RPW ? R - r0 : RPW;
                tile_regs_acquire();
                for (uint32_t rr = 0; rr < nr; ++rr) {
                    UNPACK((bsr::unpack_hold_run(
                        bsr::unpack_addr(cb_x, (r0 + rr) * WS), bsr::unpack_addr(cb_inv, r0 + rr), WS, 1)));
#ifdef TRISC_MATH
                    for (uint32_t c = 0; c < WS; ++c) {
                        bsr::math_hold_tile<true>(rr * WS + c);
                    }
                    bsr::math_hold_release_b();
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < nr * WS; ++i) {
                    pack_tile<true>(i, cb_out, r0 * WS + i);
                }
                tile_regs_release();
            }
        } else if constexpr (V == 34 || V == 35) {
            // HOLD with per-face srcA: 34 = inv (Col, row-major runs), 35 = γ (Row, column-major runs)
            constexpr bool C34 = V == 34;
            constexpr uint32_t RUNS = C34 ? R : WS;
            constexpr uint32_t RUNLEN = C34 ? WS : R;
            uint32_t slot = 0, t0 = 0;
            for (uint32_t q = 0; q < RUNS; ++q) {
                UNPACK((bsr::unpack_hold_faces_run(
                    bsr::unpack_addr(cb_x, C34 ? q * WS : q),
                    bsr::unpack_addr(C34 ? cb_inv : cb_g, q),
                    RUNLEN,
                    C34 ? 1 : WS)));
                for (uint32_t k = 0; k < RUNLEN; ++k) {
                    if (slot == 0) {
                        tile_regs_acquire();
                    }
                    MATH((bsr::math_hold_faces_tile<C34>(slot)));
                    ++slot;
                    const bool last = (q + 1 == RUNS) && (k + 1 == RUNLEN);
                    if (slot == SB || last) {
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < slot; ++i) {
                            const uint32_t tt = t0 + i;  // run-major position
                            const uint32_t oi = C34 ? tt : (tt % R) * WS + tt / R;
                            pack_tile<true>(i, cb_out, oi);
                        }
                        tile_regs_release();
                        t0 += slot;
                        slot = 0;
                    }
                }
                MATH((bsr::math_hold_release_b()));
            }
        } else if constexpr (V == 16 || V == 17 || V == 18 || V == 19) {
            // HOLD inv, row-major: per tile-row r, inv_r held in srcB for the WS tiles x[r][0..WS-1]
            // DEST windows of SB tiles span rows; srcB is released at each row end.
            uint32_t slot = 0, t0 = 0;
            for (uint32_t r = 0; r < R; ++r) {
#ifdef TRISC_UNPACK
                if constexpr (V == 19) {
                    if (r == 0) {
                        bsr::unpack_hold_run_pf(0, bsr::unpack_addr(cb_inv, 0), 0, 1, true);
                    }
                    bsr::unpack_hold_run_pf(
                        bsr::unpack_addr(cb_x, r * WS),
                        bsr::unpack_addr(cb_inv, r + 1 < R ? r + 1 : r),
                        WS,
                        1,
                        r + 1 < R);
                } else if constexpr (V == 18) {
                    mock_set(0, 1);
                    for (uint32_t c = 0; c < WS; ++c) {
                        mock_set(1, 0);
                    }
                } else {
                    bsr::unpack_hold_run(bsr::unpack_addr(cb_x, r * WS), bsr::unpack_addr(cb_inv, r), WS, 1);
                }
#endif
                for (uint32_t c = 0; c < WS; ++c) {
                    if constexpr (V == 16 || V == 19) {
                        if (slot == 0) {
                            tile_regs_acquire();
                        }
                    }
#ifdef TRISC_MATH
                    if constexpr (V == 17) {
                        mock_clr(1, 0);
                    } else {
                        bsr::math_hold_tile<true>(slot);
                    }
#endif
                    ++slot;
                    const bool last = (r + 1 == R) && (c + 1 == WS);
                    if constexpr (V == 16 || V == 19) {
                        if (slot == SB || last) {
                            tile_regs_commit();
                            tile_regs_wait();
                            for (uint32_t i = 0; i < slot; ++i) {
                                pack_tile<true>(i, cb_out, t0 + i);
                            }
                            tile_regs_release();
                            t0 += slot;
                            slot = 0;
                        }
                    } else if (slot == SB) {
                        slot = 0;
                    }
                }
#ifdef TRISC_MATH
                if constexpr (V == 17) {
                    mock_clr(0, 1);
                } else {
                    bsr::math_hold_release_b();
                }
#endif
            }
        }
        if constexpr (V == 21) {
            // drain the per-row sums (unpack thread pops what pack pushed)
            cb_wait_front(cb_sq, R);
            cb_pop_front(cb_sq, R);
        }
    }
    if constexpr (V < 21 || V >= 30) {
        cb_push_back(cb_out, NT);
    }
}
