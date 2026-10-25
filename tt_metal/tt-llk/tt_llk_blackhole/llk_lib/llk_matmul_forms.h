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
// direction and 2 the other one (in1 held in SrcA, in0 streamed through SrcB). Bit 2 (value 4) is the two-row scheme: two rows (or columns) of the
// held operand sit in the two banks of its source and every streamed tile is used for both; it keeps the default direction. A kernel fixes
// the value at compile time with matmul_reuse_dir.
template <std::uint32_t reuse_dir>
constexpr bool matmul_reuse_a(const std::uint32_t ct_dim, const std::uint32_t rt_dim)
{
    return (reuse_dir & 3) == 0 ? ct_dim >= rt_dim : (reuse_dir & 3) == 1;
}
template <std::uint32_t reuse_dir>
constexpr bool matmul_two_row()
{
    return (reuse_dir & 4) != 0;
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
// two_row_off: the blocks of the two-row scheme that measured slower than the single row one
constexpr std::uint32_t matmul_reuse_dir(
    const MathFidelity fidelity,
    const DataFormat in0,
    const DataFormat in1,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const bool pack_bound   = false,
    const bool two_row_off  = false)
{
    // a Float32 x Float32 block with an even held dimension streams each tile once per pair of held rows
    const std::uint32_t held = ct_dim >= rt_dim ? rt_dim : ct_dim;
    if (fidelity == MathFidelity::LoFi && !two_row_off && in0 == DataFormat::Float32 && in1 == DataFormat::Float32 && held >= 2 && (held % 2) == 0)
    {
        return 4;
    }
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
    {MathFidelity::HiFi3, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, false},
    {MathFidelity::HiFi3, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, false},
    {MathFidelity::HiFi3, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, false},
    {MathFidelity::HiFi4, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, true},
    {MathFidelity::LoFi, false, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, false},
    {MathFidelity::LoFi, false, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false, 7},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
    // LoFi at full sync: the row MOP of a Float32 input to Bfp8_b with a 32-bit DEST reads 2.6 cycles slower than the tile form on four row blocks; so do the Float16_b and Float32 outputs of a Float32 input (+2.6 and +0.6) on 4x2 and 2x4 blocks (1x4 and 4x1 not measured)
    {MathFidelity::LoFi, false, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false, 4, 4, 1, 2, 4},
    {MathFidelity::LoFi, false, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float16_b, false, 4, 4, 1, 2, 4, 2},
    {MathFidelity::LoFi, false, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, false, 4, 4, 1, 2, 4, 2},
};

template <MathFidelity math_fidelity, std::uint32_t kt_per_section, bool fp32_dest, DataFormat in0, DataFormat in1, DataFormat out_format, DataFormat pack_format, bool several_blocks, std::uint32_t row_tiles = 0, int sync = 0, std::uint32_t min_side = 99>
constexpr bool matmul_main_form_cell_selected()
{
    for (const matmul_main_form_cell& c : matmul_main_form_cells)
    {
        if (c.fidelity == math_fidelity && c.k1 == (kt_per_section == 1) && c.fp32_dest == fp32_dest && c.out == out_format && c.pack == pack_format &&
            (c.in == in0 || c.in == in1) && !(c.several_blocks_ok && several_blocks) && row_tiles >= c.min_row && row_tiles <= c.max_row && (c.sync < 0 || c.sync == sync) &&
            kt_per_section >= c.min_k && kt_per_section <= c.max_k && min_side >= c.min_dim)
        {
            return true;
        }
    }
    return false;
}

// Cells that take a form of their own whatever the table says, keyed on the input format and the block shape (the table keys the output
// format, which cells of different input formats share): form 'R' is the hoisted rows form, 'h' the tile MOP with the hoisted unpack. The
// kernels key a forced cell as Bfp2 ('R') or Bfp2_b ('h'), which matmul_default_form reads as that form (neither is a ttnn type).
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
};

template <MathFidelity math_fidelity, std::uint32_t kt_per_section, bool fp32_dest, DataFormat in0, DataFormat in1, DataFormat out_format, DataFormat pack_format, int sync, std::uint32_t h, std::uint32_t w, bool several_blocks = false, std::uint32_t dest_sections = 1>
constexpr char matmul_forced_form()
{
    for (const matmul_forced_form_cell& c : matmul_forced_form_cells)
    {
        if (c.fidelity == math_fidelity && c.sync == sync && kt_per_section >= c.min_k && kt_per_section <= c.max_k && c.fp32_dest == fp32_dest &&
            c.out == out_format && c.pack == pack_format && (c.in == in0 || c.in == in1) && w >= c.min_w && w <= c.max_w && h >= c.min_h && (!c.wide || w >= h) && (!c.several || several_blocks) &&
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
    return form == 'R' ? DataFormat::Bfp2 : (form == 'h' ? DataFormat::Bfp2_b : DataFormat::Invalid);
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
} // namespace ckernel
