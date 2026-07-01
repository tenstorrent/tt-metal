// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"

/**
 * Does BH unpack_tilize leave face rows interleaved in SrcA for this input format?
 *
 * ONLY meaningful in the unpack_tilize -> math_datacopy -> pack pipeline. Do not call
 * from other pack contexts; the answer is only valid when the unpacker was configured
 * for strided reads by _llk_unpack_tilize_init_.
 *
 * BH background:
 *   BH has a 128-byte L1 read interface split into 8 x 16-byte reads in tilize
 *   (strided) mode. Stride is applied only to interfaces 0, 2, 4, 6; interfaces
 *   1, 3, 5, 7 read at (paired even interface) + 16 bytes:
 *     i0 = base                     i1 = base + 16
 *     i2 = base + stride            i3 = base + stride + 16
 *     i4 = base + 2*stride          i5 = base + 2*stride + 16
 *     i6 = base + 3*stride          i7 = base + 3*stride + 16
 *   Non-8-bit formats use all 8 interfaces, so face rows land in SrcA in the wrong
 *   order and the packer must reorder them by running in PackMode::Tilize.
 *   8-bit formats use only interfaces 0, 2, 4, 6 (correctly strided) so SrcA is
 *   already in the layout pack expects and the packer runs in PackMode::Default.
 *   WH has a 64-byte / 4-interface tilize path and no bug.
 */
inline constexpr bool unpack_tilize_interleaves_rows(std::uint32_t unpack_src_format)
{
    return !IS_8BIT_FORMAT(unpack_src_format);
}
