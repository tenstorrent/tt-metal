// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"

// Default matmul unpack and math forms for the keys a kernel fixes at compile time: k steps per DEST section (one call
// per k step), math fidelity, DEST sync, 32-bit DEST and output format. Each entry is the form that, on the tt-llk
// matmul sweep, runs fastest with no configuration slower than main beyond noise: '.' per-row unpack and tile MOP,
// 'r' per-row unpack and row MOP, 'h' hoisted unpack and tile MOP, 'R' hoisted unpack and row MOP. One string per
// (k class, fidelity, sync, DEST width), one character per output format (Float16_b, Float16, Float32, Bfp8_b, Bfp4_b).
// An unknown k (0) or another output format takes '.', the forms main runs.
namespace ckernel
{
inline constexpr const char* matmul_forms_table[48] = {
    "....h", "r.r.R", "rrrrR", "..r.r", // k 1, LoFi
    ".R..R", "rr..R", "rrRrR", "rrrrr", // k 1, HiFi2
    "rr.rR", "rrrrR", "rrRrR", "rrrrr", // k 1, HiFi3
    "rR.rR", "rrrrR", "rrrrr", "rrrrr", // k 1, HiFi4
    "rh..R", "rrrrR", "RRrrR", "r.rrR", // k 2-4, LoFi
    "RRRRR", "rRrrR", "RRRRR", "rRrRR", // k 2-4, HiFi2
    "rRRrR", "rRrrR", "rRRRR", "rRRrR", // k 2-4, HiFi3
    "RRRRR", "RRRRR", "rrrrr", "rrrrr", // k 2-4, HiFi4
    "RRRRR", "RRRRR", "RRRRR", "RRrRR", // k 5+, LoFi
    "RRRRR", "RRRRR", "RRRRR", "RRRRR", // k 5+, HiFi2
    "RRRRR", "RRRRR", "RRRRR", "RRRRR", // k 5+, HiFi3
    "RRRRR", "RRRRR", "RrrRr", "rrrrr", // k 5+, HiFi4
};

// Reuse direction of a block: reuse_dir 0 holds in0 in SrcB and streams in1 through SrcA when ct_dim >= rt_dim (main's rule), 1 forces that
// direction and 2 the other one (in1 held in SrcA, in0 streamed through SrcB). A kernel fixes it at compile time with matmul_reuse_dir.
template <std::uint32_t reuse_dir>
constexpr bool matmul_reuse_a(const std::uint32_t ct_dim, const std::uint32_t rt_dim)
{
    return reuse_dir == 0 ? ct_dim >= rt_dim : reuse_dir == 1;
}

// A 32-bit tile takes twice the unpacker cycles of a 16-bit one (32 against 64 datums per cycle), and the streamed operand is unpacked once per
// output tile, the held one once per row: with one operand of 32 bits and the other of 16 bits at LoFi the block holds the 32-bit operand and
// streams the 16-bit one, which halves the unpack stream of a block of 4 tiles or more where the held operand is reused (>= 2 rows or columns).
// The default direction already does it when the 32-bit operand is in0 and ct_dim >= rt_dim, or in1 and ct_dim < rt_dim.
constexpr bool matmul_format_is_32bit(const DataFormat f)
{
    return f == DataFormat::Float32 || f == DataFormat::Tf32;
}
constexpr bool matmul_format_is_16bit(const DataFormat f)
{
    return f == DataFormat::Float16 || f == DataFormat::Float16_b;
}
// pack_bound: half sync with a Float32 output, where the pack is the bound and the cell measured up to 1.7 percent slower
constexpr std::uint32_t matmul_reuse_dir(
    const MathFidelity fidelity, const DataFormat in0, const DataFormat in1, const std::uint32_t ct_dim, const std::uint32_t rt_dim, const bool pack_bound = false)
{
    if (fidelity != MathFidelity::LoFi || pack_bound || ct_dim * rt_dim < 4)
    {
        return 0;
    }
    if (matmul_format_is_32bit(in1) && matmul_format_is_16bit(in0) && ct_dim >= rt_dim && rt_dim >= 2)
    {
        return 2;
    }
    if (matmul_format_is_32bit(in0) && matmul_format_is_16bit(in1) && ct_dim < rt_dim && ct_dim >= 2)
    {
        return 1;
    }
    return 0;
}

template <MathFidelity math_fidelity, DstSync dst_sync, bool fp32_dest, std::uint32_t kt_per_section, DataFormat out_format>
constexpr char matmul_default_form()
{
    constexpr int fmt = out_format == DataFormat::Float16_b ? 0
                        : out_format == DataFormat::Float16 ? 1
                        : out_format == DataFormat::Float32 ? 2
                        : out_format == DataFormat::Bfp8_b  ? 3
                        : out_format == DataFormat::Bfp4_b  ? 4
                                                            : -1;
    if constexpr (out_format == DataFormat::Bfp2_b)
    {
        return 'h'; // forced forms (matmul_forced_form_cells): block float 2 bit formats are not matmul outputs
    }
    else if constexpr (out_format == DataFormat::Bfp2)
    {
        return 'R';
    }
    else if constexpr (out_format == DataFormat::Bfp4)
    {
        return 'r';
    }
    else if constexpr (kt_per_section == 0 || fmt < 0)
    {
        return '.';
    }
    else
    {
        constexpr std::uint32_t k_class = kt_per_section == 1 ? 0 : (kt_per_section <= 4 ? 1 : 2);
        constexpr std::uint32_t fid     = math_fidelity == MathFidelity::LoFi    ? 0
                                          : math_fidelity == MathFidelity::HiFi2 ? 1
                                          : math_fidelity == MathFidelity::HiFi3 ? 2
                                                                                 : 3;
        return matmul_forms_table[((k_class * 4 + fid) * 2 + (dst_sync == DstSync::SyncFull ? 1 : 0)) * 2 + (fp32_dest ? 1 : 0)][fmt];
    }
}
// Half sync tile MOP cells where the HiFi phase end op can ride on the last MVMUL (end_fold): HiFi2 with a 16-bit DEST and one k step per
// section, Float16_b or Bfp8_b output. Elsewhere at half sync the earlier release of the held source read slower (unpack bound cells).
template <MathFidelity math_fidelity, DstSync dst_sync, bool fp32_dest, std::uint32_t kt_per_section, DataFormat out_format, DataFormat in0 = DataFormat::Invalid, DataFormat in1 = DataFormat::Invalid>
constexpr bool matmul_half_end_fold()
{
    // HiFi3 and HiFi4 take it on 16-bit and 32-bit operands, and on Bfp8_b by Bfp8_b into Float32: with other block float operands, and at HiFi3 with a 32-bit DEST,
    // the earlier release of the held source reads slower
    constexpr bool wide_operands = ((in0 == DataFormat::Float16_b || in0 == DataFormat::Float16 || in0 == DataFormat::Float32) &&
                                    (in1 == DataFormat::Float16_b || in1 == DataFormat::Float16 || in1 == DataFormat::Float32)) ||
                                   (in0 == DataFormat::Bfp8_b && in1 == DataFormat::Bfp8_b && out_format == DataFormat::Float32);
    return dst_sync == DstSync::SyncHalf &&
           ((math_fidelity == MathFidelity::HiFi2 && !fp32_dest && kt_per_section == 1 && (out_format == DataFormat::Float16_b || out_format == DataFormat::Bfp8_b)) ||
            ((math_fidelity == MathFidelity::HiFi4 || (math_fidelity == MathFidelity::HiFi3 && !fp32_dest)) && wide_operands));
}
// Cells where the keys of the ttnn matmul kernels (the format each DEST section is packed to, the row MOP rule) pick a form that reads
// slower than main's tile MOP and per-row unpack on the tt-llk sweep (perf_matmul_forms: every form of every perf_matmul Half cell,
// ABBA against main, no fused post flip). Those cells keep main's form. A cell is the math fidelity, whether the contraction of a section
// is one tile, a 32-bit DEST, one input format (of in0 or in1), the output format and the format the section is packed to.
// several_blocks_ok: the cell keeps the keyed form when the kernel runs several inner blocks. Bfp8_b input to Float32 output from a 16-bit DEST
// with Float16_b partials (packer_l1_acc) at HiFi3 and HiFi4 with one tile per section: the last section reads slower than main (up to +1.2 cy
// per tile) but the partial sections gain more, so two or more inner blocks are faster (ttnn, HiFi3, 4 inner blocks: 1.2 % faster than main).
struct matmul_main_form_cell
{
    MathFidelity fidelity;
    bool k1;
    bool fp32_dest;
    DataFormat in;
    DataFormat out;
    DataFormat pack;
    bool several_blocks_ok;
    std::uint32_t min_row = 0;
    std::uint32_t max_row = 0xFFFF;
    int sync = -1; // 0 half, 1 full, -1 any
    std::uint32_t min_k = 0;
    std::uint32_t max_k = 0xFFFF;
    std::uint32_t min_dim = 0; // the smaller side of the block, in tiles
    DataFormat other = DataFormat::Invalid; // when set, the operand that is not `in` must have this format (a mixed pair)
};

inline constexpr matmul_main_form_cell matmul_main_form_cells[] = {
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float16_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, false},
    {MathFidelity::HiFi3, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, true},
    {MathFidelity::HiFi3, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, false},
    // Float16_b to Bfp4_b at HiFi3 from a 32-bit DEST, one k step: the keyed forms read 0.4 cycles behind main on a single tile
    {MathFidelity::HiFi3, true, true, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, false},
    {MathFidelity::HiFi3, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, false},
    {MathFidelity::HiFi3, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, false},
    {MathFidelity::HiFi3, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, false},
    {MathFidelity::HiFi4, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, true},
    {MathFidelity::LoFi, false, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, false},
    {MathFidelity::LoFi, false, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false, 7},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
    // LoFi, half sync, 16-bit DEST, Float16_b output, a 2x2 block over four k steps with one Float32 and one Float16_b operand: the rows form reads 0.26 % slower than main's
    // on the legacy single core (ttnn ABBA); the Float32 by Float32 block of the same shape gains 3.6 % from it and keeps it
    {MathFidelity::LoFi, false, false, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float16_b, false, 2, 2, 0, 4, 4, 2, DataFormat::Float16_b},
    // LoFi at full sync: the row MOP of a Float32 input to Bfp8_b with a 32-bit DEST reads 2.6 cycles slower than the tile form on four row blocks; so do the Float16_b and Float32 outputs of a Float32 input (+2.6 and +0.6) on 4x2 and 2x4 blocks (1x4 and 4x1 not measured)
    {MathFidelity::LoFi, false, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false, 4, 4, 1, 2, 4},
    {MathFidelity::LoFi, false, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float16_b, false, 4, 4, 1, 2, 4, 2},
    {MathFidelity::LoFi, false, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, false, 4, 4, 1, 2, 4, 2},
};

// Mixed operand pairs that keep main's form (perf_matmul_mix and perf_matmul_cov2, ABBA against main). Float32 in in1 (the streamed SrcA) against a
// block float in0 at HiFi2 on blocks 4 tiles wide or more: the row MOP leaves the SrcA unpacker no bubble and reads up to 9 percent slower. Narrower
// blocks take the keyed forms, which read up to 5 percent faster, except one tile over two to four k steps at full sync, a 1 by 3 block at
// half sync over two to four k steps, a 4 by 4 block of one k step and a single tile of one k step at half sync (ct_dim 0, an unknown block, keeps main's form).
// Float32 in in0 takes the keyed forms except at half sync over one k step, where they gain nothing. Bfp8_b in in1 against Float16_b in in0
// at HiFi2, half sync, 16-bit DEST, one k step also keeps main's form.
constexpr bool matmul_block_float_format(const DataFormat f)
{
    return f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b;
}

template <MathFidelity math_fidelity, std::uint32_t kt_per_section, bool fp32_dest, DataFormat in0, DataFormat in1, DataFormat out_format, int sync, std::uint32_t ct_dim = 0, std::uint32_t rt_dim = 0>
constexpr bool matmul_mixed_main_form()
{
    constexpr bool f32_in1 = in1 == DataFormat::Float32 && matmul_block_float_format(in0);
    constexpr bool keyed =
        ct_dim != 0 &&
        (kt_per_section > 1 ? (ct_dim < 4 && !(sync == 1 && kt_per_section <= 4 && ct_dim == 1 && rt_dim == 1) && !(sync == 0 && kt_per_section <= 4 && ct_dim == 3 && rt_dim == 1))
                            : (!(ct_dim >= 4 && rt_dim >= 4) && !(sync == 0 && ct_dim == 1 && rt_dim == 1)));
    return math_fidelity == MathFidelity::HiFi2 &&
           ((f32_in1 && !keyed) || (in0 == DataFormat::Float32 && matmul_block_float_format(in1) && sync == 0 && kt_per_section == 1) ||
            (!fp32_dest && kt_per_section == 1 && sync == 0 && in1 == DataFormat::Bfp8_b && in0 == DataFormat::Float16_b && out_format == DataFormat::Float32));
}

template <MathFidelity math_fidelity, std::uint32_t kt_per_section, bool fp32_dest, DataFormat in0, DataFormat in1, DataFormat out_format, DataFormat pack_format, bool several_blocks, std::uint32_t row_tiles = 0, int sync = 0, std::uint32_t min_side = 99, std::uint32_t ct_dim = 0, std::uint32_t rt_dim = 0>
constexpr bool matmul_main_form_cell_selected()
{
    if (matmul_mixed_main_form<math_fidelity, kt_per_section, fp32_dest, in0, in1, out_format, sync, ct_dim, rt_dim>())
    {
        return true;
    }
    for (const matmul_main_form_cell& c : matmul_main_form_cells)
    {
        if (c.fidelity == math_fidelity && c.k1 == (kt_per_section == 1) && c.fp32_dest == fp32_dest && c.out == out_format && c.pack == pack_format &&
            (c.in == in0 || c.in == in1) && (c.other == DataFormat::Invalid || (c.in == in0 && in1 == c.other) || (c.in == in1 && in0 == c.other)) &&
            !(c.several_blocks_ok && several_blocks) && row_tiles >= c.min_row && row_tiles <= c.max_row && (c.sync < 0 || c.sync == sync) &&
            kt_per_section >= c.min_k && kt_per_section <= c.max_k && min_side >= c.min_dim)
        {
            return true;
        }
    }
    return false;
}

// Cells that take a form of their own whatever the table says, keyed on the input format and the block shape (the table keys the output
// format, which cells of different input formats share): form 'R' is the hoisted rows form, 'h' the tile MOP with the hoisted unpack. The
// kernels key a forced cell as Bfp2 ('R'), Bfp2_b ('h') or Bfp4 ('r'), which matmul_default_form reads as that form (none is a ttnn type).
struct matmul_forced_form_cell
{
    MathFidelity fidelity;
    int sync; // 0 half, 1 full
    std::uint32_t min_k;
    std::uint32_t max_k;
    bool fp32_dest;
    DataFormat in;
    DataFormat out;
    DataFormat pack;
    std::uint32_t min_w; // block width (ct) in tiles
    std::uint32_t max_w;
    std::uint32_t min_h; // block height (rt) in tiles
    bool wide;           // width at least the height: the in1 side is reused
    char form;
    bool several = false; // only with several inner blocks (the sweeps run one)
    int sections = 0;     // 0 any, 1 only a kernel of one DEST section (what the sweeps run), 2 only several sections
    std::uint32_t max_h = 0xFFFF; // block height (rt) in tiles, at most
    bool in0_only = false;        // the input format must be that of in0, not in1
};

inline constexpr matmul_forced_form_cell matmul_forced_form_cells[] = {
    // Float32 to Bfp8_b at LoFi, one k step, 32-bit DEST: the 2x2 block reads 1.5 cycles behind main on the per row unpack; the four tile row and
    // column blocks lose with the hoisted form
    // (one DEST section only: with 4 sections the hoisted form reads 0.7 cycles behind main on the math sweep)
    {MathFidelity::LoFi, 0, 1, 1, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 0xFFFF, 2, false, 'R', false, 1},
    // Bfp8_b to Float32 at LoFi, two to four k steps, 16-bit DEST: the rows form reads up to 5 cycles under main from a 4 tile width (the
    // tile form leaves 0.14 to 0.19 over); the 3 tile width takes the hoisted tile form; the tall blocks keep the tile form
    {MathFidelity::LoFi, 0, 2, 4, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 4, 0xFFFF, 1, true, 'R'},
    {MathFidelity::LoFi, 0, 2, 4, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 3, 3, 1, true, 'h'},
    // Bfp8_b to Float32 at HiFi4, one k step, 16-bit DEST: the hoisted tile form on the wide blocks (the tall blocks lose with it)
    // (one DEST section: the blocks the sweeps run; several sections: only from 6 tiles wide, where ttnn gains 24 percent, the narrower ones read 0.3 cycles
    // behind with 4 sections)
    {MathFidelity::HiFi4, 0, 1, 1, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 0xFFFF, 1, true, 'h', false, 1},
    {MathFidelity::HiFi4, 0, 1, 1, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 6, 0xFFFF, 1, true, 'h', false, 2},
    // Float16_b and Bfp8_b to Float32 at HiFi2 and HiFi4, one k step per inner block, 16-bit DEST: the 2x4 block (the row MOP leaves Float32 from a 16-bit
    // DEST on the tile form) reads 0.7 to 1.4 percent faster in ttnn with the hoisted rows form, 8 and 32 inner blocks
    {MathFidelity::HiFi2, 0, 1, 1, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 4, 4, 2, true, 'R', true},
    {MathFidelity::HiFi2, 0, 1, 1, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 4, 4, 2, true, 'R', true},
    {MathFidelity::HiFi4, 0, 1, 1, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 4, 4, 2, true, 'R', true},
    // The rows form without the hoisted unpack ('r'; the kernels' own row MOP stays) where the hoisted unpack of the table's form reads 0.1 to 2 cycles slower than
    // main on the tt-llk dyn harness (equal in0 and in1 format, ABBA over its four throttle modes) and the unhoisted one does not; the single core ttnn cells among
    // them (Bfp8_b HiFi2 4x2, Float16_b HiFi2 8x1 at half sync) read the same either way
    // Bfp8_b HiFi2, a 4 by 2 block over 2 to 32 k steps, at half and full sync
    {MathFidelity::HiFi2, 0, 2, 32, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 2, 4, false, 'r', false, 0, 4},
    {MathFidelity::HiFi2, 1, 2, 32, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 2, 4, false, 'r', false, 0, 4},
    // Bfp8_b to Float32 at HiFi2, half sync, one k step, 16-bit DEST, a single tile: the table's form reads 0.6 cycles behind main
    {MathFidelity::HiFi2, 0, 1, 1, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 1, 1, false, 'r', false, 0, 1, true},
    // Float16_b HiFi4 at full sync over 32 k steps, 1x1, 1x8 and 2x4 blocks; at half sync over 2 to 4 k steps, a single tile
    {MathFidelity::HiFi4, 1, 32, 32, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 1, false, 'r', false, 0, 1},
    {MathFidelity::HiFi4, 1, 32, 32, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 8, false, 'r', false, 0, 8},
    {MathFidelity::HiFi4, 1, 32, 32, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 2, 2, 4, false, 'r', false, 0, 4},
    {MathFidelity::HiFi4, 0, 2, 4, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 1, false, 'r', false, 0, 1},
    // Float16_b HiFi2 at half sync over 2 to 4 k steps, an 8 tile column; LoFi over 32 k steps, 8x1 and 4x2 blocks
    {MathFidelity::HiFi2, 0, 2, 4, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 8, false, 'r', false, 0, 8},
    {MathFidelity::LoFi, 0, 32, 32, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 8, false, 'r', false, 0, 8},
    {MathFidelity::LoFi, 0, 32, 32, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 2, 2, 4, false, 'r', false, 0, 4},
};

template <MathFidelity math_fidelity, std::uint32_t kt_per_section, bool fp32_dest, DataFormat in0, DataFormat in1, DataFormat out_format, DataFormat pack_format, int sync, std::uint32_t h, std::uint32_t w, bool several_blocks = false, std::uint32_t dest_sections = 1>
constexpr char matmul_forced_form()
{
    for (const matmul_forced_form_cell& c : matmul_forced_form_cells)
    {
        if (c.fidelity == math_fidelity && c.sync == sync && kt_per_section >= c.min_k && kt_per_section <= c.max_k && c.fp32_dest == fp32_dest &&
            c.out == out_format && c.pack == pack_format && (c.in == in0 || c.in == in1) && (!c.in0_only || c.in == in0) && w >= c.min_w && w <= c.max_w && h >= c.min_h && h <= c.max_h && (!c.wide || w >= h) && (!c.several || several_blocks) &&
            (c.sections == 0 || (c.sections == 1 ? dest_sections == 1 : dest_sections > 1)))
        {
            return c.form;
        }
    }
    return 0;
}

// The key a kernel hands the table for a forced form
constexpr DataFormat matmul_forced_form_key(const char form)
{
    return form == 'R' ? DataFormat::Bfp2 : (form == 'h' ? DataFormat::Bfp2_b : (form == 'r' ? DataFormat::Bfp4 : DataFormat::Invalid));
}


/**
 * @brief The math fidelity whose phases cover every nonzero mantissa slice of the two matmul operands.
 *
 * A phase multiplies one slice of each operand: SrcA hi {1, m[9:6]} or lo m[5:1], SrcB hi {1, m[9:4]} or lo {m[3:0], 000}
 * (tt_fpu_tile.sv). An in1 (SrcA) of Bfp4_b has 2 fraction bits, so its lo slice is zero and phase 1 adds exactly 0;
 * an in0 (SrcB) of Bfp8_b or Bfp4_b has at most 6 fraction bits, so its lo slice is zero and phase 2 adds exactly 0;
 * phase 3 needs both lo slices. Only a dropped suffix maps to an existing fidelity, so HiFi3 and HiFi4 with a narrow
 * in1 and a wide in0 keep their phases.
 *
 * Only equal operand formats are keyed; Invalid (unknown, the default of the compute API) keeps the requested fidelity.
 * check_out is set by the compute API, which knows the output format and DEST width; the harness keys that cell in python.
 *
 * @tparam math_fidelity: Requested math fidelity.
 * @tparam in0_format: L1 data format of in0 (unpacked to SrcB).
 * @tparam in1_format: L1 data format of in1 (unpacked to SrcA).
 */
template <
    MathFidelity math_fidelity,
    DataFormat in0_format,
    DataFormat in1_format,
    DataFormat out_format = DataFormat::Invalid,
    bool fp32_dest        = false,
    int throttle_level    = 0,
    bool check_out        = false>
constexpr MathFidelity matmul_phase_fidelity()
{
    constexpr bool lo_a_zero = in1_format == DataFormat::Bfp4_b;
    constexpr bool lo_b_zero = in0_format == DataFormat::Bfp8_b || in0_format == DataFormat::Bfp4_b;
    // Bfp8_b x Bfp8_b at HiFi3 and HiFi4 with a 32-bit DEST or a Float32 output (or an unknown output when check_out) is pack bound and
    // measured slower with the phases dropped, so it keeps the requested fidelity
    constexpr bool keep = check_out && in0_format == DataFormat::Bfp8_b && in1_format == DataFormat::Bfp8_b && throttle_level == 0 &&
                          (math_fidelity == MathFidelity::HiFi3 || math_fidelity == MathFidelity::HiFi4) &&
                          (fp32_dest || out_format == DataFormat::Float32 || out_format == DataFormat::Invalid);
    if constexpr (in0_format != in1_format || in0_format == DataFormat::Invalid || keep)
    {
        return math_fidelity; // mixed or unknown operand formats (bits not proven on hardware), or a cell that keeps main's fidelity
    }
    else if constexpr (lo_a_zero && lo_b_zero)
    {
        return MathFidelity::LoFi;
    }
    else if constexpr (lo_b_zero)
    {
        return math_fidelity == MathFidelity::LoFi ? MathFidelity::LoFi : MathFidelity::HiFi2;
    }
    else if constexpr (lo_a_zero)
    {
        return math_fidelity == MathFidelity::HiFi2 ? MathFidelity::LoFi : math_fidelity;
    }
    else
    {
        return math_fidelity;
    }
}
// The Z counter step of the streamed in1 tile (the harness key MATMUL_STREAM_Z, test_variant_parameters.py): at LoFi where in1 streams
// (ct_dim >= rt_dim, 4 or more output tiles) and both operands are the same dense 16-bit or Float32 format in 32x32 tiles, the unpacker steps
// the Z counter one tile per UNPACR instead of a config read and write. 4 where it gained on the tt-llk sweep (ABBA, 0 slower than the config
// step), else 0. The cells that read slower, the Float32 operands at half sync and a few half sync shapes are left out.
constexpr bool matmul_stream_z_loses(
    const bool full_sync,
    const std::uint32_t rt_dim,
    const std::uint32_t ct_dim,
    const std::uint32_t kt_dim,
    const int throttle,
    const DataFormat in_format,
    const DataFormat out_format)
{
    // (sync full, rt, ct, kt, throttle, input format, output format)
    struct Cell
    {
        bool full;
        std::uint8_t rt, ct, kt, throttle;
        DataFormat in, out;
    };
    constexpr Cell cells[] = {
        {true, 1, 6, 1, 0, DataFormat::Float32, DataFormat::Float16_b},
        {true, 1, 7, 1, 0, DataFormat::Float32, DataFormat::Bfp8_b},
        {true, 1, 7, 4, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {true, 1, 7, 4, 0, DataFormat::Float16_b, DataFormat::Float16_b},
        {true, 1, 7, 4, 0, DataFormat::Float16_b, DataFormat::Float32},
        {true, 1, 7, 4, 0, DataFormat::Float32, DataFormat::Float32},
        {true, 1, 7, 32, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {true, 1, 7, 32, 0, DataFormat::Float16_b, DataFormat::Float32},
        {false, 1, 4, 1, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {false, 1, 4, 1, 0, DataFormat::Float16_b, DataFormat::Float16_b},
        {false, 1, 4, 32, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {false, 1, 5, 1, 0, DataFormat::Float16_b, DataFormat::Float32},
        {false, 1, 6, 1, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {false, 1, 7, 32, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {false, 1, 7, 32, 0, DataFormat::Float16_b, DataFormat::Float16_b},
        {false, 1, 7, 32, 0, DataFormat::Float16_b, DataFormat::Float32},
        {false, 1, 8, 32, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {false, 2, 2, 4, 5, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {false, 2, 2, 4, 5, DataFormat::Float16_b, DataFormat::Float32},
        {false, 2, 3, 1, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {false, 2, 3, 4, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
        {false, 2, 3, 4, 0, DataFormat::Float16_b, DataFormat::Float16_b},
        {false, 2, 4, 4, 0, DataFormat::Float16_b, DataFormat::Bfp8_b},
    };
    for (const Cell& c : cells)
    {
        if (c.full == full_sync && c.rt == rt_dim && c.ct == ct_dim && c.kt == kt_dim && c.throttle == throttle && c.in == in_format && c.out == out_format)
        {
            return true;
        }
    }
    return false;
}

constexpr std::uint32_t matmul_stream_z(
    const MathFidelity fidelity,
    const bool full_sync,
    const DataFormat in0,
    const DataFormat in1,
    const DataFormat out_format,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const int throttle,
    const std::uint32_t reuse_dir,
    const bool dense_tiles)
{
    if (fidelity != MathFidelity::LoFi || !dense_tiles || reuse_dir != 0 || ct_dim < rt_dim || ct_dim * rt_dim < 4 || in0 != in1)
    {
        return 0;
    }
    if (in1 != DataFormat::Float16 && in1 != DataFormat::Float16_b && in1 != DataFormat::Float32)
    {
        return 0;
    }
    // 32-bit input at half sync loses 1 percent on 1x4 with 4 blocks
    if (in1 == DataFormat::Float32 && !full_sync)
    {
        return 0;
    }
    // at half sync the Z step loses where the unpack is not the bound: k 4 on a single row, a 2 wide block at k 1, and k 1 under a static throttle
    if (!full_sync && ((kt_dim == 4 && rt_dim == 1) || (kt_dim == 1 && ct_dim == 2) || (kt_dim == 1 && throttle == 5)))
    {
        return 0;
    }
    return matmul_stream_z_loses(full_sync, rt_dim, ct_dim, kt_dim, throttle, in1, out_format) ? 0 : 4;
}
} // namespace ckernel
