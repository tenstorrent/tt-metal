// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Cross-arch stub. WH has no unpack-tilize stride bug: all 4 tilize-mode
// unpacker interfaces apply stride correctly, so face rows land in SrcA in
// the correct order and the packer never needs to reorder them. This stub
// lets cross-arch pack code call the predicate uniformly.
inline constexpr bool unpack_tilize_interleaves_rows([[maybe_unused]] std::uint32_t unpack_src_format)
{
    return false;
}
