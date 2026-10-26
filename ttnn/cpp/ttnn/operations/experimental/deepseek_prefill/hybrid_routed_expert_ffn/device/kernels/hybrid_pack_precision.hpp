// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Packs the unified half's block-float circular buffers the way the standalone unified op does.
//
// bfp8_pack_precise is one setting per binary, and the merged binary takes the fused half's: rounded
// bfp packing, from a Float16_b packer source. The unified half alone is built with truncated packing,
// from a Bfp8 source. The only thing the flag decides is that per-CB source format, and the halves own
// disjoint CB indices, so naming the unified half's indices (HYB_APPROX_PACK_CB_MASK, set by the merge
// only when the two settings differ) is enough to give each half the packing it has on its own.
//
// Interception renames the generated pack_src_format table, which every LLK pack configure and
// reconfigure path reads. The table is generated before the kernel source, so this header has to be
// parsed before any LLK pack header: it comes FIRST in the union compute kernel.

#pragma once

#if defined(UCK_CHLKC_PACK) && defined(HYB_APPROX_PACK_CB_MASK)

#include <cstdint>

#include "data_format_derive.h"

namespace hyb_pack {

struct SrcFormatTable {
    constexpr unsigned char operator[](std::uint32_t cb) const {
        const auto dst = static_cast<DataFormat>(pack_dst_format[cb]);
        if (((static_cast<std::uint64_t>(HYB_APPROX_PACK_CB_MASK) >> cb) & 1u) && ckernel::df_is_bfp_format(dst)) {
            return static_cast<unsigned char>(ckernel::df_is_exp_b_format(dst) ? DataFormat::Bfp8_b : DataFormat::Bfp8);
        }
        return pack_src_format[cb];
    }
};

}  // namespace hyb_pack

inline constexpr hyb_pack::SrcFormatTable hyb_pack_src_format{};
#define pack_src_format hyb_pack_src_format

#endif
