// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device-free compile-time contracts for matmul_block. Kept free of any compute/LLK
// headers so it can be included both by the device-side helper (for its static_assert)
// and by a host unit test (for an automated truth-table check) — one source of truth.

namespace compute_kernel_lib {

// Validity contract for matmul_block's caller_owns_pack_target mode.
//
// Under caller_owns the helper skips its own per-K-block reserve/push/drain on the pack
// targets and packs to absolute offsets in fixed regions. With Interm as the last target,
// packer L1 accumulation stays in that region for every K-block. With plain Out as the
// last target, non-last blocks accumulate in the fixed intermediate scratch, the final
// block reloads it without FIFO waits/pops, and packs into the caller-reserved output.
//
// Both forms require TileRowMajor, whose absolute-offset pack is the only one that places
// each subblock correctly into the fixed region, and packer_l1_acc. OutWithRelu and
// OutWithUntilize remain unsupported: their final-pack lifecycle is not the plain-Out one
// audited here.
constexpr bool caller_owns_pack_target_supported(
    bool caller_owns_pack_target,
    bool is_tile_row_major,
    bool packer_l1_acc,
    bool last_block_is_interm,
    bool last_block_is_plain_out) {
    return !caller_owns_pack_target ||
           (is_tile_row_major && packer_l1_acc && (last_block_is_interm || last_block_is_plain_out));
}

}  // namespace compute_kernel_lib
