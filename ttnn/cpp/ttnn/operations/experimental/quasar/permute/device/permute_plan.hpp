// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side builder of the Quasar permute block plan (word layout: kernels/permute_plan_layout.hpp).
// Pure function of the shapes: no device or tensor types, so it can be exercised stand-alone.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt-metalium/math.hpp>

#include "ttnn/operations/experimental/quasar/permute/device/kernels/permute_plan_layout.hpp"

namespace ttnn::prim::qsr::permute_plan {

struct PlanRequest {
    std::vector<uint32_t> shape;    // input logical shape, rank 1..kMaxRank, no zero extent
    std::vector<uint32_t> dims;     // output axis k takes input axis dims[k]
    bool tile = false;              // TILE (32x32, minimal padding) or ROW_MAJOR pages
    uint32_t elem_size = 0;         // bytes per element (1, 2 or 4); 0 for block-format tiles (whole-page moves only)
    uint32_t tile_page_bytes = 0;   // TILE page size
    uint32_t pad_word = 0;          // padding element bit pattern
    uint32_t max_window_bytes = 0;  // cap of one L1 window (read or write) of one thread
    uint32_t target_blocks = 1;     // the plan tries to leave at least this many blocks (work split)
};

struct BlockPlan {
    std::vector<uint32_t> words;  // kPlanWords
    uint32_t num_blocks = 0;
    uint32_t kernel_elem_size = 0;  // element size the kernel is built for
    uint32_t in_window_bytes = 0;   // per thread
    uint32_t out_window_bytes = 0;  // per thread
    bool direct = false;
};

namespace detail {

inline uint32_t floor_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p * 2 <= v) {
        p *= 2;
    }
    return p;
}

// Page strides and axis roles of one side. `shape` is that side's own logical shape.
inline void describe_side(
    const std::vector<uint32_t>& shape, bool tile, std::vector<uint32_t>& kind, std::vector<uint32_t>& page_stride) {
    const uint32_t rank = static_cast<uint32_t>(shape.size());
    kind.assign(rank, kOuter);
    page_stride.assign(rank, 0);
    if (tile) {
        const uint32_t ht = tt::div_up(shape[rank - 2], kTileDim);
        const uint32_t wt = tt::div_up(shape[rank - 1], kTileDim);
        kind[rank - 1] = kTileCol;
        page_stride[rank - 1] = 1;
        kind[rank - 2] = kTileRow;
        page_stride[rank - 2] = wt;
        uint32_t stride = ht * wt;
        for (int32_t d = static_cast<int32_t>(rank) - 3; d >= 0; --d) {
            page_stride[d] = stride;
            stride *= shape[d];
        }
    } else {
        kind[rank - 1] = kRowElem;
        uint32_t stride = 1;
        for (int32_t d = static_cast<int32_t>(rank) - 2; d >= 0; --d) {
            page_stride[d] = stride;
            stride *= shape[d];
        }
    }
}

}  // namespace detail

// True when every output page is an input page: the page-forming dimensions (the last one of ROW_MAJOR, the
// last two of TILE) stay in place, so pages only change position.
inline bool moves_whole_pages(const std::vector<uint32_t>& dims, bool tile) {
    const uint32_t last = static_cast<uint32_t>(dims.size()) - 1;
    return dims[last] == last && (!tile || dims[last - 1] == last - 1);
}

inline BlockPlan build_block_plan(const PlanRequest& req) {
    using tt::div_up;
    const uint32_t rank = static_cast<uint32_t>(req.shape.size());
    const uint32_t last = rank - 1;

    std::vector<uint32_t> shape = req.shape;
    std::vector<uint32_t> out_shape(rank);
    std::vector<uint32_t> out_pos(rank);  // input axis d lands at output axis out_pos[d]
    for (uint32_t k = 0; k < rank; ++k) {
        out_shape[k] = shape[req.dims[k]];
        out_pos[req.dims[k]] = k;
    }

    BlockPlan plan;
    plan.direct = moves_whole_pages(req.dims, req.tile);
    // Unchanged pages are moved as bytes: one kernel build serves every data type.
    const uint32_t e = plan.direct ? 1 : req.elem_size;
    plan.kernel_elem_size = e;
    if (plan.direct && !req.tile) {
        shape[last] *= req.elem_size;
        out_shape[last] *= req.elem_size;
    }

    std::vector<uint32_t> in_kind, in_page_stride, out_kind_k, out_page_stride_k;
    detail::describe_side(shape, req.tile, in_kind, in_page_stride);
    detail::describe_side(out_shape, req.tile, out_kind_k, out_page_stride_k);
    std::vector<uint32_t> out_kind(rank), out_page_stride(rank);
    for (uint32_t d = 0; d < rank; ++d) {
        out_kind[d] = out_kind_k[out_pos[d]];
        out_page_stride[d] = out_page_stride_k[out_pos[d]];
    }

    const uint32_t page_elems = req.tile ? (plan.direct ? req.tile_page_bytes : kTileElems) : 0;
    const uint32_t max_window_elems = req.max_window_bytes / e;

    std::vector<uint32_t> chunk(rank, 1);
    uint32_t flags = 0;
    uint32_t in_slot_elems = 0;
    uint32_t out_slot_elems = 0;

    auto num_blocks_of = [&]() {
        uint64_t n = 1;
        for (uint32_t d = 0; d < rank; ++d) {
            n *= div_up(shape[d], chunk[d]);
        }
        return n;
    };
    // Several whole pages per block along the longest axis that is a page axis on both sides.
    auto batch_pages = [&](uint32_t slot_elems) {
        int32_t batch_axis = -1;
        for (uint32_t d = 0; d < rank; ++d) {
            if (in_kind[d] == kOuter && out_kind[d] == kOuter &&
                (batch_axis < 0 || shape[d] > shape[static_cast<uint32_t>(batch_axis)])) {
                batch_axis = static_cast<int32_t>(d);
            }
        }
        if (batch_axis < 0) {
            return;
        }
        const uint64_t spare = num_blocks_of() / std::max<uint32_t>(req.target_blocks, 1);
        uint32_t pages = std::min<uint32_t>(max_window_elems / slot_elems, 32);
        pages = static_cast<uint32_t>(std::min<uint64_t>(pages, spare));
        chunk[static_cast<uint32_t>(batch_axis)] =
            std::clamp<uint32_t>(pages, 1, shape[static_cast<uint32_t>(batch_axis)]);
    };

    if (!req.tile) {
        // Rows are always moved as runs: [c0, c0 + n) of the row on each side.
        flags |= kFlagInSeg | kFlagOutSeg;
        if (plan.direct) {
            // A row longer than the window moves in several runs.
            flags |= kFlagDirect;
            chunk[last] = std::min<uint32_t>(shape[last], std::max<uint32_t>(max_window_elems, 1));
            in_slot_elems = out_slot_elems = chunk[last];
            if (chunk[last] == shape[last]) {
                batch_pages(in_slot_elems);
            }
        } else {
            // Input axis x becomes the output row; the input row (axis `last`) becomes an output page axis.
            // A block reads C_x row runs of C_w elements and writes C_w row runs of C_x elements.
            const uint32_t x = req.dims[last];
            chunk[x] = std::min<uint32_t>(shape[x], std::max<uint32_t>(max_window_elems, 1));
            uint32_t cw = std::min<uint32_t>(shape[last], std::max<uint32_t>(max_window_elems / chunk[x], 1));
            chunk[last] = cw;
            const uint64_t blocks = num_blocks_of();
            if (blocks < req.target_blocks) {
                // Shorter runs give more blocks to spread, but more NoC transactions: stop at 32 elements.
                const uint32_t wanted = static_cast<uint32_t>(div_up(req.target_blocks, static_cast<uint32_t>(blocks)));
                const uint32_t spread = std::max<uint32_t>(div_up(shape[last], wanted), std::min<uint32_t>(cw, 32));
                chunk[last] = std::min(cw, spread);
            }
            in_slot_elems = chunk[last];
            out_slot_elems = chunk[x];
        }
    } else {
        const uint32_t row = last - 1;
        const uint32_t col = last;
        if (plan.direct) {
            flags |= kFlagDirect;
            chunk[row] = chunk[col] = kTileDim;
            in_slot_elems = out_slot_elems = page_elems;
            batch_pages(page_elems);
        } else {
            // Output tiles are written whole, so the two axes that form the output tile are taken 32 at a time.
            flags |= kFlagFill;
            const uint32_t a = req.dims[row];
            const uint32_t b = req.dims[col];
            chunk[a] = chunk[b] = kTileDim;
            out_slot_elems = kTileElems;
            const bool row_stays = (a == row || b == row);
            const bool col_stays = (a == col || b == col);
            if (row_stays && col_stays) {
                // Transpose inside each tile.
                in_slot_elems = kTileElems;
                batch_pages(kTileElems);
            } else if (row_stays || col_stays) {
                // One input tile axis leaves the tile: 32 whole input tiles (along the axis that enters) feed
                // one output tile per index of the leaving axis.
                const uint32_t leaving = row_stays ? col : row;
                in_slot_elems = kTileElems;
                chunk[leaving] = std::clamp<uint32_t>(
                    detail::floor_pow2(std::max<uint32_t>(max_window_elems / kTileElems, 1)), 1, kTileDim);
            } else {
                // Both input tile axes leave: every element of an output tile comes from a different input tile.
                // A block reads one 16-element face line from each of 32x32 input tiles and writes 16 tiles.
                flags |= kFlagInSeg;
                chunk[row] = 1;
                chunk[col] = 16;
                in_slot_elems = 16;
            }
        }
    }

    for (uint32_t d = 0; d < rank; ++d) {
        chunk[d] = std::min(chunk[d], shape[d]);
    }

    // Window slot strides: mixed radix over the page-enumerating axes of each side.
    std::vector<uint32_t> in_slot_stride(rank, 0), out_slot_stride(rank, 0);
    uint32_t in_slots = 1;
    uint32_t out_slots = 1;
    for (int32_t d = static_cast<int32_t>(rank) - 1; d >= 0; --d) {
        if (chunk[d] > 1 && in_kind[d] == kOuter) {
            in_slot_stride[d] = in_slots;
            in_slots *= chunk[d];
        }
        if (chunk[d] > 1 && out_kind[d] == kOuter) {
            out_slot_stride[d] = out_slots;
            out_slots *= chunk[d];
        }
    }

    uint32_t inner = 0;
    for (uint32_t d = 0; d < rank; ++d) {
        if (chunk[d] > chunk[inner]) {
            inner = d;
        }
    }

    auto window_bytes = [&](uint32_t slots, uint32_t slot_elems) {
        return (slots * slot_elems * e + 63u) & ~63u;  // keeps the per-thread windows 64 B aligned
    };
    plan.in_window_bytes = window_bytes(in_slots, in_slot_elems);
    plan.out_window_bytes = plan.direct ? 0 : window_bytes(out_slots, out_slot_elems);
    plan.num_blocks = static_cast<uint32_t>(num_blocks_of());

    plan.words.assign(kPlanWords, 0);
    plan.words[kRank] = rank;
    plan.words[kFlags] = flags;
    plan.words[kInSlotElems] = in_slot_elems;
    plan.words[kOutSlotElems] = out_slot_elems;
    plan.words[kInPageElems] = page_elems;
    plan.words[kOutPageElems] = page_elems;
    plan.words[kInnerAxis] = inner;
    plan.words[kPadWord] = req.pad_word;
    plan.words[kInBufBytes] = plan.in_window_bytes;
    plan.words[kOutBufBytes] = plan.out_window_bytes;
    for (uint32_t d = 0; d < rank; ++d) {
        const uint32_t a = kHeaderWords + d * kAxisWords;
        plan.words[a + kAxisExtent] = shape[d];
        plan.words[a + kAxisChunk] = chunk[d];
        plan.words[a + kAxisInKind] = in_kind[d];
        plan.words[a + kAxisInPageStride] = in_page_stride[d];
        plan.words[a + kAxisInSlotStride] = in_slot_stride[d];
        plan.words[a + kAxisOutKind] = out_kind[d];
        plan.words[a + kAxisOutPageStride] = out_page_stride[d];
        plan.words[a + kAxisOutSlotStride] = out_slot_stride[d];
    }
    return plan;
}

}  // namespace ttnn::prim::qsr::permute_plan
