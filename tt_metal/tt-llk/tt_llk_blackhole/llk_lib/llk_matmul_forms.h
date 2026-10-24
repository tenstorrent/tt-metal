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

template <MathFidelity math_fidelity, DstSync dst_sync, bool fp32_dest, std::uint32_t kt_per_section, DataFormat out_format>
constexpr char matmul_default_form()
{
    constexpr int fmt = out_format == DataFormat::Float16_b ? 0
                        : out_format == DataFormat::Float16 ? 1
                        : out_format == DataFormat::Float32 ? 2
                        : out_format == DataFormat::Bfp8_b  ? 3
                        : out_format == DataFormat::Bfp4_b  ? 4
                                                            : -1;
    if constexpr (kt_per_section == 0 || fmt < 0)
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
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
};

template <MathFidelity math_fidelity, std::uint32_t kt_per_section, bool fp32_dest, DataFormat in0, DataFormat in1, DataFormat out_format, DataFormat pack_format, bool several_blocks>
constexpr bool matmul_main_form_cell_selected()
{
    for (const matmul_main_form_cell& c : matmul_main_form_cells)
    {
        if (c.fidelity == math_fidelity && c.k1 == (kt_per_section == 1) && c.fp32_dest == fp32_dest && c.out == out_format && c.pack == pack_format &&
            (c.in == in0 || c.in == in1) && !(c.several_blocks_ok && several_blocks))
        {
            return true;
        }
    }
    return false;
}
} // namespace ckernel
