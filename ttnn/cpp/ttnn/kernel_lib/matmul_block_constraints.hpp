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
// target: the caller does ONE reserve before the K-loop and ONE push after, and each
// K-block packs to absolute offsets in that fixed region with packer_l1_acc accumulating
// in place. That is only correct when the helper's software-reload accumulation path (the
// per-K-block spill push paired with the reload wait_front at the top of the K-loop) is
// statically dead — otherwise the reload wait_front, which is NOT gated by caller_owns,
// has no matching spill push (that push IS gated off) and the helper deadlocks.
//
// The reload path is dead iff last_block_target == Interm (the only branch that forces
// enable_reload = false). It additionally requires TileRowMajor, whose absolute-offset
// pack is the only one that places each subblock correctly into the caller's fixed region
// (SubblockMajor would not deadlock but would corrupt output), and packer_l1_acc, since
// in-place accumulation is the whole point of the fixed region. Hence: caller_owns is
// supported iff TileRowMajor + packer_l1_acc + Interm.
constexpr bool caller_owns_pack_target_supported(
    bool caller_owns_pack_target, bool is_tile_row_major, bool packer_l1_acc, bool last_block_is_interm) {
    return !caller_owns_pack_target || (is_tile_row_major && packer_l1_acc && last_block_is_interm);
}

}  // namespace compute_kernel_lib
