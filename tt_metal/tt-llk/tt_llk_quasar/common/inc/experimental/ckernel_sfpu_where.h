// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// Programs ADDR_MOD_6 with dest.incr=2 so the SFPSTORE in the replayed
// per-row body auto-advances the dest counter by one SFP row pair per
// iteration. Quasar's shared SFPU init only programs ADDR_MOD_7 (incr=0);
// where needs an addrmod that advances dest. The slot is otherwise unused
// on Quasar, so this is additive.
inline void _init_where_()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
}

// Per-lane select: out = (cond == 0) ? false_val : true_val.
// Offsets are in SFPU dest_reg_addr units (rows * 2) and address into the
// per-tile sub-region of DEST; the dest counter (advanced by ADDR_MOD_6 in
// the SFPSTORE) supplies the per-iteration row stride, so the offsets are
// constants captured verbatim by REPLAY.
//
// TODO: SFPLOADMACRO-based fast path. BH has a special case for the
// in-place form (out == in0) that schedules through SFPLOADMACRO templates
// programmed in `_init_where_` via TTI_SFPCONFIG, replaying 3 instructions
// per row pair instead of 6. The instruction exists on Quasar but no SFPU
// kernel programs these templates yet — landing it here would require
// verifying the BH SFPCONFIG bit encodings (simple_bits / store_bits / misc
// 0x770) against Quasar's SFPU template/macro register layout. Defer until
// either another Quasar SFPU kernel establishes a reference for this
// pattern, or codegen confirms it as the long-term direction for ternary
// SFPU kernels on Quasar.
inline void _calculate_where_(const int iterations, const int in0_offset_idx, const int in1_offset_idx, const int in2_offset_idx, const int out_offset_idx)
{
    // Record the 6-instruction body once into replay slots 0..5; the loop
    // below issues a REPLAY per row pair. ADDR_MOD_6's dest.incr=2 (programmed
    // in _init_where_) handles row stride across iterations.
    lltt::record(0, 6);
    TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in0_offset_idx); // condition -> LREG0
    TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in1_offset_idx); // true_val  -> LREG1
    TTI_SFPSETCC(0, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_EQ0);                      // CC := (LREG0 == 0)
    TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in2_offset_idx); // false_val -> LREG1 only on CC-enabled lanes
    TTI_SFPENCC(0, 0);                                                                 // re-enable all lanes
    TT_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0, out_offset_idx); // store + advance dest by 2

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        lltt::replay(0, 6);
    }
}

} // namespace sfpu
} // namespace ckernel
