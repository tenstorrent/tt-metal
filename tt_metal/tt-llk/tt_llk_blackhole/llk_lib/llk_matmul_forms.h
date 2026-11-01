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
    if constexpr (kt_per_section == 0)
    {
        return '.'; // main's form, also for the output formats that key a forced form below
    }
    else if constexpr (out_format == DataFormat::Bfp2_b)
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
// Half sync tile MOP cells where the HiFi phase end op can ride on the last MVMUL (end_fold): HiFi4, and HiFi3 with a 16-bit DEST except Bfp8_b in0 against
// Float16_b in1 into Float32 and, on main's form (unknown k), a Bfp4_b in1 into Float32, both when the operand formats are given; HiFi2 with a 16-bit DEST and one k step per section, Float16_b or Bfp8_b output.
// At LoFi, at HiFi3 with a 32-bit DEST and elsewhere at HiFi2 the earlier release of the held source read slower.
template <MathFidelity math_fidelity, DstSync dst_sync, bool fp32_dest, std::uint32_t kt_per_section, DataFormat out_format, DataFormat in0 = DataFormat::Invalid, DataFormat in1 = DataFormat::Invalid>
constexpr bool matmul_half_end_fold()
{
    constexpr bool known_operands = in0 != DataFormat::Invalid && in1 != DataFormat::Invalid;
    return dst_sync == DstSync::SyncHalf &&
           ((math_fidelity == MathFidelity::HiFi2 && !fp32_dest && kt_per_section == 1 && (out_format == DataFormat::Float16_b || out_format == DataFormat::Bfp8_b)) ||
            (known_operands && math_fidelity == MathFidelity::HiFi4) ||
            (known_operands && math_fidelity == MathFidelity::HiFi3 && !fp32_dest &&
             !(in0 == DataFormat::Bfp8_b && in1 == DataFormat::Float16_b && out_format == DataFormat::Float32) &&
             !(kt_per_section == 0 && in1 == DataFormat::Bfp4_b && out_format == DataFormat::Float32)));
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
    // LoFi, half sync, 16-bit DEST cells of a Float16_b or Float32 in1 against a block float in0 that read 0.2 to 11 cycles behind main with the keyed forms
    constexpr bool lofi_cell =
        math_fidelity == MathFidelity::LoFi && sync == 0 && !fp32_dest && matmul_block_float_format(in0) && ct_dim != 0 &&
        ((in1 == DataFormat::Float16_b && in0 == DataFormat::Bfp8_b && out_format == DataFormat::Float32 && rt_dim == 1 && ct_dim == 5 && (kt_per_section == 4 || kt_per_section == 32)) ||
         (in1 == DataFormat::Float32 && in0 == DataFormat::Bfp8_b && out_format == DataFormat::Float32 && rt_dim == 1 && ct_dim == 6 && kt_per_section == 32) ||
         (in1 == DataFormat::Float32 && in0 == DataFormat::Bfp4_b && out_format == DataFormat::Bfp4_b && rt_dim == 2 && ct_dim == 4 && kt_per_section == 1) ||
         (in1 == DataFormat::Float32 && in0 == DataFormat::Bfp4_b && out_format == DataFormat::Float32 && rt_dim == 1 && ct_dim == 2 && kt_per_section == 4) ||
         (in1 == DataFormat::Float32 && in0 == DataFormat::Bfp8_b && out_format == DataFormat::Bfp4_b && rt_dim == 4 && ct_dim == 2 && kt_per_section == 1));
    return lofi_cell || (math_fidelity == MathFidelity::HiFi2 &&
           ((f32_in1 && !keyed) || (in0 == DataFormat::Float32 && matmul_block_float_format(in1) && sync == 0 && kt_per_section == 1) ||
            (!fp32_dest && kt_per_section == 1 && sync == 0 && in1 == DataFormat::Bfp8_b && in0 == DataFormat::Float16_b && out_format == DataFormat::Float32)));
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

// Cells of the perf_matmul_grid sweep (every out_subblock h x w at k 1, 3 and 8 steps, one long k, the packer L1 accumulate block and static throttle) where the keyed forms
// read slower than main's: they keep main's form. A cell is the fidelity, half sync, 32-bit DEST accumulate, in1, in0 and output format, the block (rt x ct), the k steps of the
// section, the static throttle level and whether the packer L1 accumulates.
struct matmul_grid_main_cell
{
    MathFidelity fidelity;
    bool half_sync;
    bool fp32_acc;
    DataFormat in1;
    DataFormat in0;
    DataFormat out;
    std::uint8_t rt, ct, kt, throttle;
    bool l1_acc;
    std::uint8_t nb;
};
inline constexpr matmul_grid_main_cell matmul_grid_main_cells[] = {
    {MathFidelity::HiFi2, false, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 1, 3, 0, false, 1},
    {MathFidelity::HiFi2, false, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 6, 2, 1, 0, true, 1},
    {MathFidelity::HiFi2, false, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 1, 3, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 1, 1, 0, true, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 3, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 3, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 3, 3, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 4, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 3, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 3, 3, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 4, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 1, 3, 8, 0, false, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 1, 1, 0, true, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 1, 1, 1, false, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 2, 3, 0, false, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 2, 8, 0, false, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 1, 3, 8, 0, false, 1},
    {MathFidelity::HiFi3, true, false, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi3, true, false, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi3, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi3, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 8, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 6, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 7, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 8, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3, 3, 0, true, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 3, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 8, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 8, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 5, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 5, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 6, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 7, 8, 0, true, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 8, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 8, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 3, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 5, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 7, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 5, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 5, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 7, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 8, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 8, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 5, 3, 4, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 7, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 3, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 2, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 3, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 5, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 6, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 7, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 7, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 8, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 2, 2, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 2, 3, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 2, 4, 8, 1, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 3, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 5, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 6, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 5, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 5, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 6, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3, 8, 1, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 2, 2, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 1, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 2, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 2, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 2, 3, 1, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 4, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 5, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 6, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 7, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1, 8, 4, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 1, 7, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 7, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 4, 2, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 5, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 7, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 7, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 3, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 3, 2, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 2, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 2, 3, 4, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 3, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 2, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 2, 3, 4, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 3, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 4, 8, 0, true, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 1, 1, 1, 5, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 2, 2, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 3, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 2, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 3, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 1, 1, 1, 5, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3, 8, 1, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1, 3, 1, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 3, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 4, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 2, 1, 4, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 4, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float16_b, 1, 3, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float16_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, 1, 2, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, 1, 4, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 1, 3, 8, 1, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 2, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 2, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float16_b, 1, 4, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 1, 2, 3, 0, false, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 1, 1, 0, true, 4},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 1, 4, 3, 0, true, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1, 1, 0, true, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 1, 1, 0, true, 2},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 1, 1, 0, true, 2},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 1, 0, true, 2},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 3, 1, 1, 0, true, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 1, 0, true, 2},
    {MathFidelity::HiFi3, true, true, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 3, 1, 1, 0, true, 4},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 7, 32, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 7, 32, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 7, 32, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 7, 32, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 7, 32, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 7, 32, 0, false, 1},
    {MathFidelity::LoFi, false, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 1, 5, 1, 0, true, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 2, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 8, 0, true, 2},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 5, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 6, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 7, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 7, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 7, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 8, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 8, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 8, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 2, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 5, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 5, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 7, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 3, 2, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 3, 2, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 6, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 6, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 7, 1, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 2, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 6, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 6, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 7, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 7, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 8, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 8, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 6, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 6, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 8, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 8, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 6, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 6, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 8, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 8, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 7, 3, 0, true, 4},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 4, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 5, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 7, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 4, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 5, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 1, 8, 0, true, 4},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 5, 1, 0, true, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 6, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 7, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 8, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 6, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 7, 3, 0, true, 2},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 6, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 6, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 6, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 4, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 3, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 3, 2, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 2, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 3, 2, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 3, 2, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1, 1, 0, true, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 8, 0, true, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 1, 0, true, 2},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 2, 1, 8, 0, true, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 3, 1, 8, 0, true, 4},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1, 1, 0, true, 4},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 8, 0, true, 2},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 2, 8, 0, true, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3, 8, 0, true, 4},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 1, 8, 0, true, 4},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 4, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 4, 8, 0, true, 2},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 4, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float16_b, 1, 4, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 2, 2, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 3, 0, true, 4},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1, 32, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 4, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 4, 1, 32, 0, false, 1},
    // LLK-only space (formats and throttles the grid sweep does not reach) where the keyed forms read slower than main's
    {MathFidelity::LoFi, true, false, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp2_b, 4, 2, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp2_b, DataFormat::Bfp4_b, 1, 4, 1, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp2_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1, 8, 2, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1, 1, 2, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1, 8, 3, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp2_b, 2, 1, 8, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp2_b, 4, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 2, 8, 2, false, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float16, 1, 4, 8, 0, false, 1},
    // LoFi, 32-bit DEST accumulate, a 1x4 block of a Float16 in1 against a Bfp8 in0 to Bfp8 over 3 k steps: the keyed forms and the closed block pack read 3.8 cycles behind main, main's form
    // with the per-tile pack (matmul_pack_exact_per_tile_cells) reads level with it; LoFi, 16-bit DEST, a 2x4 block of a Float16_b in1 against a Bfp4_b in0 to Bfp8_b over 8 k steps at throttle 2: 0.6 behind, main's form and per-tile pack -0.03
    {MathFidelity::LoFi, true, true, DataFormat::Float16, DataFormat::Bfp8, DataFormat::Bfp8, 1, 4, 3, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 4, 8, 2, false, 1},
    // Full sync 1x1 blocks of Float16_b or Bfp8_b: re-initialised before every call, the keyed forms read up to 10 cycles per call behind main's
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 1, 1, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 4, 0, false, 1},
    {MathFidelity::LoFi, false, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 1, 4, 0, false, 1},
    {MathFidelity::HiFi2, false, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi2, false, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 1, 1, 0, false, 1},
    {MathFidelity::HiFi2, false, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 4, 0, false, 1},
    {MathFidelity::HiFi2, false, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 1, 4, 0, false, 1},
    {MathFidelity::HiFi2, false, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 32, 0, false, 1},
    {MathFidelity::HiFi2, false, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 1, 32, 0, false, 1},
    {MathFidelity::HiFi4, false, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 4, 0, false, 1},
    {MathFidelity::HiFi4, false, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 1, 4, 0, false, 1},
    {MathFidelity::HiFi4, false, false, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 1, 32, 0, false, 1},
    {MathFidelity::HiFi4, false, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 1, 32, 0, false, 1},
};

constexpr bool matmul_grid_main_form(const MathFidelity fidelity, const bool half_sync, const bool fp32_acc, const DataFormat in1, const DataFormat in0, const DataFormat out,
                                     const std::uint32_t rt, const std::uint32_t ct, const std::uint32_t kt, const std::uint32_t throttle = 0, const bool l1_acc = false, const std::uint32_t nb = 1)
{
    for (const matmul_grid_main_cell& c : matmul_grid_main_cells)
    {
        if (c.fidelity == fidelity && c.half_sync == half_sync && c.fp32_acc == fp32_acc && c.in1 == in1 && c.in0 == in0 && c.out == out && c.rt == rt && c.ct == ct && c.kt == kt &&
            c.throttle == throttle && c.l1_acc == l1_acc && c.nb == nb)
        {
            return true;
        }
    }
    return false;
}


// Half sync cells with packer L1 accumulate where the fused section post (the post and the DEST flip in one stall) holds the packer longer than main's separate post.
inline constexpr matmul_grid_main_cell matmul_unfused_post_cells[] = {
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1, 1, 0, true, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 2, 1, 8, 0, true, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 6, 8, 0, true, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 2, 8, 0, true, 1},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 3, 1, 1, 0, true, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 1, 4, 3, 0, true, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 3, 1, 8, 0, true, 4},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3, 8, 0, true, 4},
    {MathFidelity::LoFi, true, true, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1, 1, 0, true, 2},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 1, 1, 0, false, 1},
};

// The 1x2 block of one K tile at LoFi from a Float32 in1 (the A key) and a Bfp8_b in0 (the B key) to Float16_b of full tiles: the per-tile pack reads 0.7 cycles ahead of the block run.
constexpr bool matmul_pack_per_tile_cell(const MathFidelity fidelity, const bool half_sync, const std::uint32_t throttle, const DataFormat in1, const DataFormat in0, const DataFormat out,
                                         const std::uint32_t rt, const std::uint32_t ct, const std::uint32_t kt, const std::uint32_t r0 = 32, const std::uint32_t c1 = 32)
{
    return fidelity == MathFidelity::LoFi && half_sync && throttle == 0 && in1 == DataFormat::Float32 && in0 == DataFormat::Bfp8_b && out == DataFormat::Float16_b && rt == 1 && ct == 2 &&
           kt == 1 && r0 == 32 && c1 == 32;
}

// Blocks where the table's pack reads behind main's per-tile pack at one exact k (the cells of matmul_grid_main_cells that keep main's form): the per-tile pack
struct matmul_pack_exact_cell
{
    MathFidelity fidelity;
    std::uint8_t throttle;
    DataFormat in_a;
    DataFormat in_b;
    DataFormat out;
    std::uint8_t rt, ct, kt;
};
inline constexpr matmul_pack_exact_cell matmul_pack_exact_per_tile_cells[] = {
    {MathFidelity::LoFi, 0, DataFormat::Float16, DataFormat::Bfp8, DataFormat::Bfp8, 1, 4, 3},
    {MathFidelity::LoFi, 2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 4, 8},
};

constexpr bool matmul_pack_exact_per_tile(const MathFidelity fidelity, const std::uint32_t throttle, const DataFormat in_a, const DataFormat in_b, const DataFormat out, const std::uint32_t rt,
                                          const std::uint32_t ct, const std::uint32_t kt)
{
    for (const matmul_pack_exact_cell& c : matmul_pack_exact_per_tile_cells)
    {
        if (c.fidelity == fidelity && c.throttle == throttle && c.in_a == in_a && c.in_b == in_b && c.out == out && c.rt == rt && c.ct == ct && c.kt == kt)
        {
            return true;
        }
    }
    return false;
}

constexpr bool matmul_format_block_float(const DataFormat f)
{
    return f == DataFormat::Bfp8 || f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b || f == DataFormat::Bfp2_b;
}

struct matmul_pack_block_cell
{
    MathFidelity fidelity;
    DataFormat out;
    DataFormat in_a;
    DataFormat in_b;
    std::uint8_t rt, ct;
};
// Blocks of three or more K tiles at LoFi and HiFi2 where the closed block pack read slower than the per-tile pack in the tt-llk sweep (any K, accumulate and packer L1 accumulate).
inline constexpr matmul_pack_block_cell matmul_pack_per_tile_cells[] = {
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 5, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 8, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 4, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 4, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 5, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 7, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 7, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 1, 7},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 1, 8},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 2, 3},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 2, 4},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 1, 7},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 1, 8},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 2, 3},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 7, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 4, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float32, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 2, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 4, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 3, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 4, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float32, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float32, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 4},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 3},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 1, 8},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 2, 1},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 4},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 3, 1},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 3, 2},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 7, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 5, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 8, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 6, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 5},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 6},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 8},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 7, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 8, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 5},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 6},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float16_b, 7, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 1, 5},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 1, 6},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 7, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 8, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 4},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 6},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 7},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 8},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 6},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 7},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 8},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 4},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 7, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 1, 8},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 2, 4},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 1, 4},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 1, 8},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 7, 1},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 8, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 3},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 4},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 7},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 7, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 8, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, 1, 4},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, 1, 5},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, 2, 4},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 1, 4},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 1, 8},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 2, 3},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 2, 4},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 2, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 3, 1},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 8, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 4},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 3, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 5},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 7},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 8},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 8, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 2, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float16_b, 7, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float32, 1, 4},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float32, 1, 6},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float32, 1, 8},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float32, 4, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 4},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 6},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 7, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 8, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 2, 4},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 5, 1},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 4},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 6, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 3, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 8, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 8, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 7},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 7, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 8, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float16_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 1, 4},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 3},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 2, 4},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 4, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float16_b, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float16_b, 1, 3},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float16_b, 1, 4},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float16_b, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 1, 3},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 1, 4},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 3, 1},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 8, 1},
};

constexpr bool matmul_pack_grid_format(const DataFormat f)
{
    return f == DataFormat::Float16_b || f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b || f == DataFormat::Float32;
}

constexpr bool matmul_pack_per_tile_block(const MathFidelity fidelity, const DataFormat out, const DataFormat in_a, const DataFormat in_b, const std::uint32_t rt, const std::uint32_t ct)
{
    for (const matmul_pack_block_cell& c : matmul_pack_per_tile_cells)
    {
        if (c.fidelity == fidelity && c.out == out && c.in_a == in_a && c.in_b == in_b && c.rt == rt && c.ct == ct)
        {
            return true;
        }
    }
    return false;
}

struct matmul_pack_llkx_cell
{
    MathFidelity fidelity;
    DataFormat out;
    DataFormat in_a;
    DataFormat in_b;
    std::uint8_t rt, ct, throttle;
};
// Cells of the LLK-only space (Float16, Bfp8 and Bfp2_b operands, throttle 2 and 3) where the block pack read slower than the per-tile pack
// (any K and accumulate): the other blocks of the sweep take the block pack, which read faster on 3400 of 4100 of them.
inline constexpr matmul_pack_llkx_cell matmul_pack_llkx_per_tile_cells[] = {
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Bfp2_b, DataFormat::Bfp2_b, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Bfp2_b, DataFormat::Bfp4_b, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Bfp2_b, DataFormat::Bfp8_b, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Bfp4_b, DataFormat::Bfp2_b, 3, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Float16_b, DataFormat::Bfp2_b, 3, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Float32, DataFormat::Bfp2_b, 1, 8, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 2, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp2_b, DataFormat::Float32, DataFormat::Float16_b, 3, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 4, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 2, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 1, 5, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 2, 2, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8, DataFormat::Float16, DataFormat::Float16, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp8, DataFormat::Float16, DataFormat::Float16, 3, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp2_b, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 8, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 2, 4, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp2_b, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 2, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 2, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float16_b, 2, 2, 2},
    {MathFidelity::HiFi2, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 1, 5, 2},
    {MathFidelity::HiFi2, DataFormat::Float16, DataFormat::Bfp8, DataFormat::Bfp8, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16, DataFormat::Bfp8, DataFormat::Float16, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16, DataFormat::Float16, DataFormat::Bfp8, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16, DataFormat::Float16, DataFormat::Float16, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16, DataFormat::Float16_b, DataFormat::Float16_b, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16, DataFormat::Float32, DataFormat::Float32, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp2_b, DataFormat::Bfp8_b, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 2, 3, 2},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8, DataFormat::Bfp8, 1, 3, 0},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8, DataFormat::Bfp8, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Bfp8, DataFormat::Float16, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float16, DataFormat::Bfp8, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float16, DataFormat::Float16, 2, 1, 0},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 3},
    {MathFidelity::HiFi2, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 1, 2, 3},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 6, 2},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 4, 2},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 2, 2},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 1, 8, 2},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp2_b, 1, 6, 0},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 5, 1, 2},
    {MathFidelity::HiFi2, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 5, 1, 3},
    {MathFidelity::HiFi3, DataFormat::Bfp2_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 3, 0},
    {MathFidelity::HiFi3, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Float32, 2, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 2, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 4, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 7, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 7, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 1, 8, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 4, 1, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 2, 1, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 1, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float16_b, 7, 1, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, 1, 4, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 2, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, 2, 1, 2},
    {MathFidelity::HiFi3, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 5, 2},
    {MathFidelity::HiFi3, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 4, 1, 2},
    {MathFidelity::HiFi3, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, 1, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp2_b, 1, 2, 0},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float32, 2, 1, 3},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Bfp8, DataFormat::Bfp8, 2, 1, 0},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Bfp8, DataFormat::Float16, 2, 1, 0},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 2, 2, 2},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 2, 3, 2},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Float16, DataFormat::Bfp8, 2, 1, 0},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Float16, DataFormat::Float16, 2, 1, 0},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp2_b, 1, 2, 0},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float16_b, 1, 2, 3},
    {MathFidelity::HiFi3, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 2, 3, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 2, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 2, 2, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 2, 2, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float16_b, 7, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Float32, 3, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 4, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Float32, 4, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 3, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 2, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 5, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 2, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 5, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, 1, 4, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 4, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 4, 2},
    {MathFidelity::HiFi4, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 2, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 5, 2},
    {MathFidelity::HiFi4, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 5, 2},
    {MathFidelity::HiFi4, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float32, 2, 3, 2},
    {MathFidelity::HiFi4, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 2, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 1, 6, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 1, 2, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, 2, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 4, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 6, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 4, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float16_b, 1, 2, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, 2, 2, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp8_b, 2, 1, 2},
    {MathFidelity::HiFi4, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 2, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Bfp2_b, DataFormat::Bfp4_b, 1, 8, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Bfp2_b, DataFormat::Bfp4_b, 2, 4, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Bfp2_b, DataFormat::Bfp8_b, 1, 6, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Bfp2_b, DataFormat::Float16_b, 2, 4, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Bfp2_b, DataFormat::Float32, 2, 4, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Float16_b, DataFormat::Bfp2_b, 1, 6, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 3, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 6, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Float16_b, DataFormat::Float32, 1, 6, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Float32, DataFormat::Bfp4_b, 1, 4, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Float32, DataFormat::Float16_b, 2, 1, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Float32, DataFormat::Float32, 1, 2, 0},
    {MathFidelity::LoFi, DataFormat::Bfp2_b, DataFormat::Float32, DataFormat::Float32, 2, 2, 0},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp2_b, DataFormat::Float32, 2, 3, 0},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Bfp2_b, 4, 2, 0},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 2, 1, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp2_b, 1, 3, 0},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 3, 1, 3},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float16_b, DataFormat::Float32, 2, 1, 2},
    {MathFidelity::LoFi, DataFormat::Bfp4_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 8, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8, DataFormat::Bfp8, DataFormat::Float16, 4, 1, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8, DataFormat::Float16, DataFormat::Bfp8, 1, 7, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8, DataFormat::Float16, DataFormat::Bfp8, 4, 1, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8, DataFormat::Float16, DataFormat::Bfp8, 6, 1, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp4_b, 1, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 1, 3, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 4, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 2, 2, 2},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 2, 2, 3},
    {MathFidelity::LoFi, DataFormat::Float16, DataFormat::Bfp8, DataFormat::Float16, 2, 1, 0},
    {MathFidelity::LoFi, DataFormat::Float16, DataFormat::Bfp8, DataFormat::Float16, 3, 1, 0},
    {MathFidelity::LoFi, DataFormat::Float16, DataFormat::Bfp8, DataFormat::Float16, 6, 1, 0},
    {MathFidelity::LoFi, DataFormat::Float16, DataFormat::Float16, DataFormat::Bfp8, 1, 7, 0},
    {MathFidelity::LoFi, DataFormat::Float16, DataFormat::Float16, DataFormat::Bfp8, 1, 8, 0},
    {MathFidelity::LoFi, DataFormat::Float16, DataFormat::Float32, DataFormat::Float32, 4, 1, 0},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp2_b, 4, 2, 0},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, 1, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16, DataFormat::Bfp8, 1, 8, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 1, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Bfp2_b, 2, 1, 0},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 1, 2, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Bfp8_b, 2, 3, 3},
    {MathFidelity::LoFi, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, 2, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp2_b, DataFormat::Bfp4_b, 2, 3, 0},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp2_b, DataFormat::Bfp8_b, 1, 4, 0},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp2_b, DataFormat::Float16_b, 1, 6, 0},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 1, 2, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8, DataFormat::Float16, 6, 1, 0},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp2_b, 4, 1, 0},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 1, 8, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, 2, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16, DataFormat::Bfp8, 1, 3, 0},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16, DataFormat::Bfp8, 1, 8, 0},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16, DataFormat::Float16, 1, 8, 0},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 2, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, 2, 1, 2},
    {MathFidelity::LoFi, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 1, 5, 3},
};

constexpr bool matmul_pack_llkx_format(const DataFormat f)
{
    return f == DataFormat::Float16 || f == DataFormat::Float16_b || f == DataFormat::Float32 || f == DataFormat::Bfp8 || f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b || f == DataFormat::Bfp2_b;
}

// The LLK-only space: a Float16, Bfp8 or Bfp2_b operand or output, or throttle 2 or 3 (the grid sweep covers the other blocks, which keep matmul_pack_form_base_k).
constexpr bool matmul_pack_llkx_space(const DataFormat in_a, const DataFormat in_b, const DataFormat out, const std::uint32_t throttle)
{
    const auto grid = [](const DataFormat f) { return f == DataFormat::Float16_b || f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b || f == DataFormat::Float32; };
    return throttle == 2 || throttle == 3 || !grid(in_a) || !grid(in_b) || !grid(out);
}

constexpr bool matmul_pack_llkx_per_tile(const MathFidelity fidelity, const DataFormat out, const DataFormat in_a, const DataFormat in_b, const std::uint32_t rt, const std::uint32_t ct,
                                         const std::uint32_t throttle)
{
    for (const matmul_pack_llkx_cell& c : matmul_pack_llkx_per_tile_cells)
    {
        if (c.fidelity == fidelity && c.out == out && c.in_a == in_a && c.in_b == in_b && c.rt == rt && c.ct == ct && c.throttle == throttle)
        {
            return true;
        }
    }
    return false;
}

// Output pack form of a matmul block: 0 per tile, 1 one block run, 2 the closed block (padded pages), 4 the dense run of small tiles (dense_pages only: the perf sweep, where the
// output address does not change the pack timing). in_a and in_b are the A and B operand formats, r0 and c1 the in0 tile height and in1 tile width. The one decision of the
// harness (matmul_pack_form of matmul_sweep.py mirrors it) and of the keyed llk_matmul_block_pack.
constexpr int matmul_pack_form_base_k(const MathFidelity fidelity, const bool half_sync, const std::uint32_t throttle, const std::uint32_t kt, const std::uint32_t rt, const std::uint32_t ct,
                                 const DataFormat in_a, const DataFormat in_b, const DataFormat out, const std::uint32_t r0 = 32, const std::uint32_t c1 = 32, const bool dense_pages = false,
                                 const bool l1_acc = false)
{
    const bool small = r0 < 32 || c1 < 32;
    const bool block_float_out = matmul_format_block_float(out);
    // the packer's L1 accumulate of a Float32 output reads and writes 64 byte rows: at half sync the block run reads slower than the per-tile pack
    if (l1_acc && half_sync && out == DataFormat::Float32)
    {
        return 0;
    }
    if (rt * ct < 2)
    {
        return 0;
    }
    // LoFi and HiFi2 blocks of three or more K tiles take the closed block, which reads faster than the per-tile pack, except in the cells of matmul_pack_per_tile_cells
    if ((fidelity == MathFidelity::LoFi || fidelity == MathFidelity::HiFi2) && half_sync && throttle == 0 && kt >= 3 && !small && matmul_pack_grid_format(in_a) && matmul_pack_grid_format(in_b) &&
        matmul_pack_grid_format(out) && !matmul_pack_per_tile_block(fidelity, out, in_a, in_b, rt, ct))
    {
        return 2;
    }
    if (fidelity == MathFidelity::LoFi && kt > 1 && half_sync)
    {
        return 0;
    }
    if (matmul_pack_per_tile_cell(fidelity, half_sync, throttle, in_a, in_b, out, rt, ct, kt, r0, c1))
    {
        return 0;
    }
    // HiFi2 two-tile blocks of one K tile (not to a block float output), and one-row Bfp8_b blocks of five or more tiles from Float32 operands: the per-tile pack is on the critical path
    if (half_sync && throttle == 0 && !small && out != DataFormat::Float32 && (in_a == DataFormat::Float16_b || in_a == DataFormat::Float32) && fidelity == MathFidelity::HiFi2 &&
        ((kt == 1 && rt * ct == 2 && !block_float_out) || (in_a == DataFormat::Float32 && out == DataFormat::Bfp8_b && rt == 1 && ct >= 5)))
    {
        return block_float_out ? 2 : 1;
    }
    // small tiles packed to Float32 behind Float16_b operands: the closed block beats the per-tile pack at LoFi
    if (half_sync && throttle == 0 && fidelity == MathFidelity::LoFi && kt == 1 && small && out == DataFormat::Float32 && in_a == DataFormat::Float16_b && ct >= 4)
    {
        return 2;
    }
    // small tiles packed to Float32 behind Bfp8_b operands: the closed block beats the per-tile pack at LoFi and HiFi2 half sync
    if (half_sync && throttle == 0 && kt == 1 && (fidelity == MathFidelity::LoFi || fidelity == MathFidelity::HiFi2) && small && in_a == DataFormat::Bfp8_b && out == DataFormat::Float32 &&
        rt * ct >= 4)
    {
        return 2;
    }
    // a Bfp4_b output of two or more full tiles takes the closed block pack at half sync too, except against a Float32 operand
    if (out == DataFormat::Bfp4_b && !small && in_a != DataFormat::Float32 && in_b != DataFormat::Float32)
    {
        return 2;
    }
    // half sync at HiFi2: one K tile of small tiles, or with a Float32 output, leaves the per-tile pack on the critical path
    const bool half_block = kt == 1 && fidelity == MathFidelity::HiFi2 && (small || out == DataFormat::Float32);
    if (half_sync && ((fidelity != MathFidelity::LoFi && !half_block) || throttle >= 4 || (small && out == DataFormat::Float32) ||
                      ((in_a == DataFormat::Float32 || (in_b == DataFormat::Float32 && out == DataFormat::Bfp4_b)) && out != DataFormat::Float32 && rt * ct >= 3)))
    {
        return 0;
    }
    if (dense_pages && small && c1 == 32 && !block_float_out)
    {
        return (half_sync && rt * ct >= 8) ? 2 : 4;
    }
    return (small || block_float_out) ? 2 : 1;
}

// Blocks of the math only sweep (dense_pages: the perf sweep of math without unpack) where the pack form of the table reads behind main's per-tile pack: the form of the cell instead
// (0 per tile; the block run of a block float output writes wrong data), keyed on the operand and output formats, the block and the k steps (any accumulate).
struct matmul_pack_dense_cell
{
    MathFidelity fidelity;
    DataFormat out;
    DataFormat in_a;
    DataFormat in_b;
    std::uint8_t rt, ct, kt;
    std::uint8_t form;
};
inline constexpr matmul_pack_dense_cell matmul_pack_dense_cells[] = {
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 4, 2, 4, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float16_b, 1, 4, 4, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float16_b, 2, 4, 4, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 3, 1, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 3, 3, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 3, 4, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 4, 1, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 1, 4, 4, 0},
    {MathFidelity::LoFi, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, 4, 1, 4, 0},
};

// -1 where the table's form stays
constexpr int matmul_pack_dense_form(const MathFidelity fidelity, const std::uint32_t kt, const std::uint32_t rt, const std::uint32_t ct, const DataFormat in_a, const DataFormat in_b,
                                     const DataFormat out)
{
    for (const matmul_pack_dense_cell& c : matmul_pack_dense_cells)
    {
        if (c.fidelity == fidelity && c.out == out && c.in_a == in_a && c.in_b == in_b && c.rt == rt && c.ct == ct && c.kt == kt)
        {
            return c.form;
        }
    }
    return -1;
}

// The output pack form of a matmul block: matmul_pack_form_base_k, and for the blocks it leaves on the per-tile pack at half sync (two or more full tiles, throttle 0, 2 or 3, no packer L1 accumulate,
// Float and block float formats) the block pack of the LLK-only sweep: the plain block run, the closed block for a block float output, except the cells of matmul_pack_llkx_per_tile_cells.
constexpr int matmul_pack_form_k(const MathFidelity fidelity, const bool half_sync, const std::uint32_t throttle, const std::uint32_t kt, const std::uint32_t rt, const std::uint32_t ct,
                                 const DataFormat in_a, const DataFormat in_b, const DataFormat out, const std::uint32_t r0 = 32, const std::uint32_t c1 = 32, const bool dense_pages = false,
                                 const bool l1_acc = false)
{
    if (half_sync && r0 == 32 && c1 == 32 && !l1_acc && matmul_pack_exact_per_tile(fidelity, throttle, in_a, in_b, out, rt, ct, kt))
    {
        return 0;
    }
    if (dense_pages && half_sync && r0 == 32 && c1 == 32 && !l1_acc && throttle == 0)
    {
        const int dense = matmul_pack_dense_form(fidelity, kt, rt, ct, in_a, in_b, out);
        if (dense >= 0)
        {
            return dense;
        }
    }
    int form = matmul_pack_form_base_k(fidelity, half_sync, throttle, kt, rt, ct, in_a, in_b, out, r0, c1, dense_pages, l1_acc);
    if (form == 0 && half_sync && rt * ct >= 2 && r0 == 32 && c1 == 32 && !l1_acc && (throttle == 0 || throttle == 2 || throttle == 3) && matmul_pack_llkx_format(in_a) && matmul_pack_llkx_format(in_b) &&
        matmul_pack_llkx_format(out) && matmul_pack_llkx_space(in_a, in_b, out, throttle) && !matmul_pack_llkx_per_tile(fidelity, out, in_a, in_b, rt, ct, throttle))
    {
        form = matmul_format_block_float(out) ? 2 : 1;
    }
    return form;
}


// Half sync single K tile blocks of Float32 output from Bfp8 operands (5-bit exponent) where the half sync end fold of the HiFi phase end reads slower than main's separate phase end.
inline constexpr matmul_grid_main_cell matmul_end_fold_off_cells[] = {
    {MathFidelity::HiFi3, true, false, DataFormat::Bfp8, DataFormat::Bfp8, DataFormat::Float32, 1, 8, 1, 0, false, 1},
    {MathFidelity::HiFi3, true, false, DataFormat::Bfp8, DataFormat::Bfp8, DataFormat::Float32, 3, 1, 1, 0, false, 1},
    {MathFidelity::HiFi3, true, false, DataFormat::Bfp8, DataFormat::Bfp8, DataFormat::Float32, 8, 1, 1, 0, false, 1},
    {MathFidelity::HiFi3, true, false, DataFormat::Bfp8, DataFormat::Float16, DataFormat::Float32, 1, 8, 1, 0, false, 1},
    {MathFidelity::HiFi4, true, false, DataFormat::Bfp8, DataFormat::Bfp8, DataFormat::Float32, 1, 8, 1, 0, false, 1},
    {MathFidelity::HiFi4, true, false, DataFormat::Bfp8, DataFormat::Bfp8, DataFormat::Float32, 8, 1, 1, 0, false, 1},
    {MathFidelity::HiFi4, true, false, DataFormat::Bfp8, DataFormat::Float16, DataFormat::Float32, 1, 8, 1, 0, false, 1},
};

constexpr bool matmul_end_fold_off(const MathFidelity fidelity, const bool half_sync, const bool fp32_acc, const DataFormat in1, const DataFormat in0, const DataFormat out, const std::uint32_t rt,
                                   const std::uint32_t ct, const std::uint32_t kt, const std::uint32_t throttle = 0, const bool l1_acc = false, const std::uint32_t nb = 1)
{
    for (const matmul_grid_main_cell& c : matmul_end_fold_off_cells)
    {
        if (c.fidelity == fidelity && c.half_sync == half_sync && c.fp32_acc == fp32_acc && c.in1 == in1 && c.in0 == in0 && c.out == out && c.rt == rt && c.ct == ct && c.kt == kt &&
            c.throttle == throttle && c.l1_acc == l1_acc && c.nb == nb)
        {
            return true;
        }
    }
    return false;
}

// Cells where the half sync DEST flip drops its base write (the next op sets it) instead of keeping it, which the keep rule below applies to every Bfp8_b output at LoFi and HiFi2: the
// 2x4 block of a Float16_b in1 against a Bfp4_b in0 at throttle 2 reads 0.5 cycles faster without the write.
inline constexpr matmul_grid_main_cell matmul_keep_flip_off_cells[] = {
    {MathFidelity::LoFi, true, false, DataFormat::Float16_b, DataFormat::Bfp4_b, DataFormat::Bfp8_b, 2, 4, 8, 2, false, 1},
};

constexpr bool matmul_keep_flip_off(const MathFidelity fidelity, const bool half_sync, const bool fp32_acc, const DataFormat in1, const DataFormat in0, const DataFormat out, const std::uint32_t rt,
                                    const std::uint32_t ct, const std::uint32_t kt, const std::uint32_t throttle = 0, const bool l1_acc = false, const std::uint32_t nb = 1)
{
    for (const matmul_grid_main_cell& c : matmul_keep_flip_off_cells)
    {
        if (c.fidelity == fidelity && c.half_sync == half_sync && c.fp32_acc == fp32_acc && c.in1 == in1 && c.in0 == in0 && c.out == out && c.rt == rt && c.ct == ct && c.kt == kt &&
            c.throttle == throttle && c.l1_acc == l1_acc && c.nb == nb)
        {
            return true;
        }
    }
    return false;
}

// Whether the half sync DEST flip of a block keeps its base write (tile_regs_commit keep_flip_write): a Bfp8_b output below HiFi3 reads slower without it, except the cells above.
constexpr bool matmul_keep_flip_write(const MathFidelity fidelity, const bool half_sync, const bool fp32_acc, const DataFormat in1, const DataFormat in0, const DataFormat out, const std::uint32_t rt,
                                      const std::uint32_t ct, const std::uint32_t kt, const std::uint32_t throttle = 0, const bool l1_acc = false, const std::uint32_t nb = 1)
{
    return out == DataFormat::Bfp8_b && (fidelity == MathFidelity::LoFi || fidelity == MathFidelity::HiFi2) &&
           !matmul_keep_flip_off(fidelity, half_sync, fp32_acc, in1, in0, out, rt, ct, kt, throttle, l1_acc, nb);
}

// matmul_half_end_fold for a block whose shape is known: the cells of matmul_end_fold_off_cells skip the fold.
template <MathFidelity math_fidelity, DstSync dst_sync, bool fp32_dest, std::uint32_t kt_per_section, DataFormat out_format, DataFormat in0, DataFormat in1, std::uint32_t rt_dim, std::uint32_t ct_dim,
          std::uint32_t kt_dim>
constexpr bool matmul_half_end_fold_block()
{
    return matmul_half_end_fold<math_fidelity, dst_sync, fp32_dest, kt_per_section, out_format, in0, in1>() &&
           !matmul_end_fold_off(math_fidelity, dst_sync == DstSync::SyncHalf, fp32_dest, in1, in0, out_format, rt_dim, ct_dim, kt_dim);
}

// Half sync cells that keep main's whole section end: the separate post and the DEST flip with its base write (the fused post and the skipped base write read slower).
inline constexpr matmul_grid_main_cell matmul_main_flip_cells[] = {
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 1, 7, 8, 0, false, 1},
    {MathFidelity::LoFi, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Float16_b, 1, 7, 3, 0, false, 1},
};

constexpr bool matmul_main_flip(const MathFidelity fidelity, const bool half_sync, const bool fp32_acc, const DataFormat in1, const DataFormat in0, const DataFormat out,
                                const std::uint32_t rt, const std::uint32_t ct, const std::uint32_t kt, const std::uint32_t throttle = 0, const bool l1_acc = false,
                                const std::uint32_t nb = 1)
{
    for (const matmul_grid_main_cell& c : matmul_main_flip_cells)
    {
        if (c.fidelity == fidelity && c.half_sync == half_sync && c.fp32_acc == fp32_acc && c.in1 == in1 && c.in0 == in0 && c.out == out && c.rt == rt && c.ct == ct && c.kt == kt &&
            c.throttle == throttle && c.l1_acc == l1_acc && c.nb == nb)
        {
            return true;
        }
    }
    return false;
}

constexpr bool matmul_unfused_post(const MathFidelity fidelity, const bool half_sync, const bool fp32_acc, const DataFormat in1, const DataFormat in0, const DataFormat out,
                                   const std::uint32_t rt, const std::uint32_t ct, const std::uint32_t kt, const std::uint32_t throttle = 0, const bool l1_acc = false,
                                   const std::uint32_t nb = 1)
{
    for (const matmul_grid_main_cell& c : matmul_unfused_post_cells)
    {
        if (c.fidelity == fidelity && c.half_sync == half_sync && c.fp32_acc == fp32_acc && c.in1 == in1 && c.in0 == in0 && c.out == out && c.rt == rt && c.ct == ct && c.kt == kt &&
            c.throttle == throttle && c.l1_acc == l1_acc && c.nb == nb)
        {
            return true;
        }
    }
    return false;
}

// Tiny tile cells (in0 height under 32) where the keyed forms and the tiny module's extra keys read slower than main's: they keep main's unpack and math form.
struct matmul_tiny_main_cell
{
    MathFidelity fidelity;
    bool half_sync;
    bool fp32_acc;
    DataFormat in1;
    DataFormat in0;
    DataFormat out;
    std::uint8_t r0, c1, rt, ct, kt;
};
inline constexpr matmul_tiny_main_cell matmul_tiny_main_cells[] = {
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 4, 32, 1, 4, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 32, 1, 4, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 8, 32, 1, 4, 4},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Bfp4_b, 4, 32, 1, 4, 4},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 4, 32, 1, 4, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 16, 32, 1, 8, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp4_b, 4, 32, 2, 2, 4},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp4_b, 8, 32, 1, 4, 1},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 16, 32, 1, 8, 1},
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp4_b, DataFormat::Bfp4_b, 4, 32, 2, 2, 4},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, 8, 32, 2, 2, 4},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, 4, 32, 1, 4, 4},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, 2, 32, 2, 2, 4},
};

constexpr bool matmul_tiny_main_form(const MathFidelity fidelity, const bool half_sync, const bool fp32_acc, const DataFormat in1, const DataFormat in0, const DataFormat out,
                                     const std::uint32_t r0, const std::uint32_t c1, const std::uint32_t rt, const std::uint32_t ct, const std::uint32_t kt)
{
    for (const matmul_tiny_main_cell& c : matmul_tiny_main_cells)
    {
        if (c.fidelity == fidelity && c.half_sync == half_sync && c.fp32_acc == fp32_acc && c.in1 == in1 && c.in0 == in0 && c.out == out && c.r0 == r0 && c.c1 == c1 && c.rt == rt &&
            c.ct == ct && c.kt == kt)
        {
            return true;
        }
    }
    return false;
}

// The k key a block runs its forms by: 0, main's form, where the block is one of the grid cells above; the caller's k steps per section otherwise.
template <MathFidelity math_fidelity, int sync, bool dest_acc, std::uint32_t kt_per_section, std::uint32_t kt_dim, DataFormat in0, DataFormat in1, DataFormat out_format,
          std::uint32_t ct_dim, std::uint32_t rt_dim, std::uint32_t throttle = 0, bool l1_acc = false, std::uint32_t nb = 1, std::uint32_t in0_r_dim = 32, std::uint32_t in1_c_dim = 32>
constexpr std::uint32_t matmul_form_k()
{
    return (matmul_grid_main_form(math_fidelity, sync == 0, dest_acc, in1, in0, out_format, rt_dim, ct_dim, kt_dim, throttle, l1_acc, nb) ||
            matmul_tiny_main_form(math_fidelity, sync == 0, dest_acc, in1, in0, out_format, in0_r_dim, in1_c_dim, rt_dim, ct_dim, kt_dim))
               ? 0
               : kt_per_section;
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
    // LoFi, 16-bit DEST, one DEST section, a 1x8 block of a Float16_b in1 against a Float32 in0 over 8 k steps to Bfp8_b (static throttle 3): the table's form reads 0.28 cycles behind main, the hoisted tile form 0.38 ahead
    {MathFidelity::LoFi, 0, 8, 8, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, 8, 8, 1, true, 'h', false, 1, 1, true},
    // LoFi, 16-bit DEST, one DEST section, a 1x7 block over 8 k steps of a Bfp8 in0 under a Float16 in1 to a Bfp8 output: the opt-in row MOP reads 27 cycles behind main at half sync
    // and 10 at full sync (the 1x6 and 1x8 blocks gain from it); the hoisted rows form reads 4.9 cycles ahead of main at half sync and 37 at full sync
    {MathFidelity::LoFi, 0, 8, 8, false, DataFormat::Bfp8, DataFormat::Bfp8, DataFormat::Bfp8, 7, 7, 1, true, 'R', false, 1, 1, true},
    {MathFidelity::LoFi, 1, 8, 8, false, DataFormat::Bfp8, DataFormat::Bfp8, DataFormat::Bfp8, 7, 7, 1, true, 'R', false, 1, 1, true},
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

// The forced form that replaces the row MOP a caller opts into (the row_mop argument of the math calls), as the harness runs it: 0 where the row MOP stays. At half sync every
// forced cell takes the place of the row MOP, at full sync the hoisted forms ('R', 'h'). h, w and kt_dim are the caller's block keys (rt, ct and k steps of the block).
template <MathFidelity math_fidelity, bool fp32_dest, DataFormat in0, DataFormat in1, DataFormat out_format, int sync, std::uint32_t h, std::uint32_t w, std::uint32_t kt_dim, std::uint32_t kt_per_section = 0>
constexpr char matmul_row_mop_form()
{
    constexpr std::uint32_t sections = (kt_per_section == 0 || kt_per_section >= kt_dim) ? 1 : 2;
    constexpr char form              = matmul_forced_form<math_fidelity, kt_dim, fp32_dest, in0, in1, out_format, out_format, sync, h, w, false, sections>();
    return (form != 0 && (sync == 0 || form == 'R' || form == 'h')) ? form : 0;
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
