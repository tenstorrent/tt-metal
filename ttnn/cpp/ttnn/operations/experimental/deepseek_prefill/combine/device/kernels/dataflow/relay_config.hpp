// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ============================================================================
// USE_RELAY — dedicated relay ("R") tensix core per sender battery
// ============================================================================
//
// A core that runs both DM0 and DM1 kernels must use opposite NOCs for them, so the combine sender
// (which runs reader_combine + writer_combine) cannot issue its eth write on NOC_0 while reader_combine
// also uses that core. Splitting the untilizer->sender->eth chain into untilizer->sender->relay->eth
// moves the fabric send onto a dedicated single-kernel core (the relay), which is then free to use NOC_0
// for the eth write, without perturbing the sender/untilizer NOC assignments.
//
// With USE_RELAY:
//   - each sender battery gets a relay core prepended (row layout R-S-U-U-U-U), placed at explicit
//     physical rows (see combine_program_factory);
//   - writer_combine (the sender) stops opening a fabric connection; it forwards each token (route_info +
//     payload) into the relay's c_24 receive ring over NOC, credit-flow-controlled;
//   - writer_relay (the relay) is the fabric endpoint: it opens the eth connections, runs the cross-chip
//     init/exit handshake, drains its c_24 ring, and forwards each token to fabric on NOC_0.
//
// Constraints of the current implementation: FABRIC_1D only and exactly 2 senders (num_links >= 2). The
// factory TT_FATALs otherwise. Set to 0 to restore the byte-for-byte original sender->eth path.
// HOST-FACTORY toggle -> needs a libttnn rebuild.
#define USE_RELAY 1

// Slot count of the relay's c_24 L1 receive ring. Each slot holds one token = routing metadata
// (l1_alignment bytes) + one aligned output page (~14 KB), mirroring the sender's c_3 layout. The sender
// writes slots round-robin and the relay drains them in order; RELAY_SLOTS is the max number of tokens
// the sender may run ahead of the relay (credit-flow-controlled).
#define RELAY_SLOTS 64
