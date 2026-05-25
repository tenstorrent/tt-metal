// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Shared between conv3d_program_factory (host) and writer.cpp (kernel). The kernel receives
// the role as a uint32_t runtime arg and casts back to this enum; cast direction is host →
// kernel via SetRuntimeArgs/get_arg_val. The fixed underlying type makes the wire format
// stable.
//
// Single weight-share strategy enum used as a compile-time arg (`weight_share_mode`).
enum class WeightShareMode : uint32_t {
    Disabled = 0,  // single-core group: each active core reads its own weight slice from DRAM
    Chain = 1,     // per-group chain forwarding (SDPA-style hop chain)
    Mcast = 2,     // per-group hardware multicast over a row-strip bbox
};

// Per-core role within the weight-share machinery. `Local` is also used for cores that have
// no work or whose group is single-core; the kernel's outer (c_in, c_out) loop simply does
// not execute for those.
enum class WeightShareRole : uint32_t {
    Local = 0,          // no participation: read DRAM directly (or empty work — kernel exits)
    ChainInjector = 1,  // chain head: read DRAM, forward to successor
    ChainMiddle = 2,    // chain middle: receive from pred, forward to succ
    ChainTail = 3,      // chain tail: receive from pred only
    McastSender = 4,    // mcast sender: read DRAM, multicast over bbox
    McastReceiver = 5,  // mcast active receiver: ack sender, wait VALID, has compute work
    McastPassive = 6,   // mcast passive: in bbox but no work; participates only in the handshake
};
