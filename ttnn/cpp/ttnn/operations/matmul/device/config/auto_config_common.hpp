// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/tile.hpp>

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

// Helpers shared by the selector's parts (internal to config/)
namespace ttnn::operations::matmul::auto_config::detail {

constexpr uint32_t TILE_DIM = 32;

inline uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Each of `parts` cores' share of N (in B's tiles). An output tile wider than B's spans several B tiles, and each
// core's columns must fill whole output tiles.
inline uint32_t split_n(const MatmulDesc& p, uint32_t parts) {
    const uint32_t ratio = p.out_tile_w / p.in1_tile_w;
    return div_up(div_up(p.Nt, parts), ratio) * ratio;
}

inline uint32_t tile_bytes(tt::DataFormat format, uint32_t h, uint32_t w) {
    return tt::tt_metal::Tile({h, w}).get_tile_size(format);
}
// Bytes of one A, B and output (or partials) tile. The factories size the output CBs with an
// in0_tile_h x in1_tile_w tile whatever the output tensor's tile.
inline uint32_t in0_tile_bytes(const MatmulDesc& p) { return tile_bytes(p.in0_format, p.in0_tile_h, TILE_DIM); }
inline uint32_t in1_tile_bytes(const MatmulDesc& p) { return tile_bytes(p.in1_format, TILE_DIM, p.in1_tile_w); }
inline uint32_t out_tile_bytes(const MatmulDesc& p, tt::DataFormat format) {
    return tile_bytes(format, p.in0_tile_h, p.in1_tile_w);
}

inline bool is_block_float(tt::DataFormat format) {
    switch (format) {
        case tt::DataFormat::Bfp8:
        case tt::DataFormat::Bfp8_b:
        case tt::DataFormat::Bfp4:
        case tt::DataFormat::Bfp4_b:
        case tt::DataFormat::Bfp2:
        case tt::DataFormat::Bfp2_b: return true;
        default: return false;
    }
}

// The mcast kernels can't take block-float B with A tiles shorter than 16 rows; Reuse can, with K in a single
// block (it computes wrong values when it splits K for those)
inline bool needs_single_k_reuse(const MatmulDesc& p) { return is_block_float(p.in1_format) && p.in0_tile_h < 16; }

// The configs' worker cores on a sub-device (its rectangle); the factories otherwise start at (0, 0)
inline std::optional<CoreRange> pinned_workers(const HardwareDesc& hw) {
    if (!hw.pinned_origin) {
        return std::nullopt;
    }
    return CoreRange(hw.origin, CoreCoord(hw.origin.x + hw.grid.x - 1, hw.origin.y + hw.grid.y - 1));
}

// A batch of one against a batched B: only 1D in1-mcast can run it, keeping the core's rows of A resident
// in L1 and looping over B's batches (in0 reuse).
inline bool broadcasts_a(const MatmulDesc& p) { return p.batch_a == 1 && p.batch_b > 1; }

// B sharded in DRAM (width or ND): the 2D factory reads it in place; the other factories can't.
inline bool dram_sharded_b(const MatmulDesc& p) {
    return p.b.dram_sharded() && (p.b.layout == MemoryLayout::WidthSharded || p.b.layout == MemoryLayout::NdSharded);
}

// Whether a sharded tensor fixes the family (a DRAM-sharded B doesn't: the interleaved candidates handle it)
inline bool sharded_layout(const MatmulDesc& p) { return p.a.sharded() || p.b.l1_sharded() || p.out.sharded(); }

// The 1D factories can't run it: a global CB without a gather config, or a DRAM-sharded B
inline bool no_mcast_1d(const MatmulDesc& p) { return p.global_cb || dram_sharded_b(p); }

// Rows of output tiles the mcast families split across cores: all batches when fused, else one batch.
inline uint32_t output_rows(const MatmulDesc& p, bool fuse_batch) { return fuse_batch ? p.batch_a * p.Mt : p.Mt; }

// Cores with work: one per output block (a Reuse block is a slice of a batch matrix, at most one per core)
inline uint32_t cores_used(
    const MatmulDesc& p, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch) {
    if (family == Family::Reuse) {
        const uint32_t blocks = p.batch_a * p.Mt / b.per_core_M;
        return std::min(blocks, static_cast<uint32_t>(hw.grid.x * hw.grid.y));
    }
    return div_up(output_rows(p, fuse_batch), b.per_core_M) * div_up(p.Nt, b.per_core_N);
}

// Divisors of n, largest first.
inline std::vector<uint32_t> divisors_desc(uint32_t n) {
    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    for (uint32_t d = 1; d * d <= n; ++d) {
        if (n % d == 0) {
            small.push_back(d);
            if (d != n / d) {
                large.push_back(n / d);
            }
        }
    }
    // large holds n/1, n/2, ... (descending); append the small ones in descending order
    large.insert(large.end(), small.rbegin(), small.rend());
    return large;
}

// Tiles held in the destination register for one subblock. Smaller tiles don't raise this: validation's
// tile-area dest count admits more of them, but subblocks above 8 tiles of 16-row tiles compute wrong values.
inline uint32_t max_subblock_area(const MatmulDesc& p, Family family) {
    uint32_t area = p.dst_full_sync_en ? 16 : 8;
    if (p.fp32_dest_acc_en) {
        area /= 2;
        // The reuse factory caps fp32-accumulating subblocks at 4 even with full-sync dest
        if (family == Family::Reuse) {
            area = std::min(area, 4u);
        }
    }
    return area;
}

}  // namespace ttnn::operations::matmul::auto_config::detail
