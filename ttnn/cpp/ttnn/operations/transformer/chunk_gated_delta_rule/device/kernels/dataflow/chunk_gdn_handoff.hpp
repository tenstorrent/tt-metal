// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Constants of the fused chunk_gdn producer -> receiver hand-off that host factories and device kernels must
// agree on. Plain constexpr only: this header is compiled into the host library (both program factories) and
// JIT-compiled into every GDN kernel, so a drift on either side fails to compile instead of hanging or corrupting.
// Protocol specification: chunk_gdn_handoff_protocol.md, two directories up.
namespace gdn_handoff {

// The seven hand-off CBs: prep's OUTPUT index == scan's INPUT index (one physical CB per tensor on the
// producer/receiver core union).
constexpr uint32_t kCbTinv = 13;
constexpr uint32_t kCbVbeta = 14;
constexpr uint32_t kCbNkd = 18;
constexpr uint32_t kCbQdecay = 19;
constexpr uint32_t kCbIntra = 20;
constexpr uint32_t kCbDl = 22;
constexpr uint32_t kCbKdecT = 24;
// The u/mask CB: kMaskTiles WY-inverse quadrant masks pushed once by the prep reader, then (fused program only) one
// tile of producer-side credit words credit[h][slot].
constexpr uint32_t kCbU = 17;
constexpr uint32_t kMaskTiles = 3;

// Fused-program semaphore ids: ready (unused by the fused variant, kept for the shared scan reader's trailing-arg
// layout), init, then one VALID flag per hand-off slot at kSemValid + slot.
constexpr uint32_t kFusedSemReady = 0;
constexpr uint32_t kFusedSemInit = 1;
constexpr uint32_t kFusedSemValid = 2;
constexpr uint32_t kMaxSemaphores = 16;  // mirrors tt::tt_metal::NUM_SEMAPHORES (host impl constant)

// Debug trace (handoff_checks + watcher): one word per protocol step in the watcher's ring buffer, stage:4 | chunk:12 |
// slot:4 | value:12, so a hang dump shows the last steps of each core. Stages 14 / 15 are the bounded-wait timeouts
// (C9).
enum HandoffTraceStage : uint32_t {
    kTxCreditSeen = 1,
    kTxBarrierDone = 2,
    kTxValidSent = 3,
    kRxIssued = 4,
    kRxValidSeen = 5,
    kTxCreditTimeout = 14,
    kRxValidTimeout = 15,
};
constexpr uint32_t handoff_trace_word(uint32_t stage, uint32_t chunk, uint32_t slot, uint32_t value) {
    return (stage << 28) | ((chunk & 0xFFFu) << 16) | ((slot & 0xFu) << 12) | (value & 0xFFFu);
}
// Polls before a hand-off wait is declared dead (C9): ~0.2 s on the RISC, >1000x the longest legitimate wait.
constexpr uint32_t kHandoffSpinLimit = 10000000;

// Fault injection (ChunkGdnFusedProgramConfig::handoff_fault, handoff_checks builds only): one deliberate breach of
// the protocol per value, so the fault test can prove the named check fires. The kernels see it as the
// GDN_HANDOFF_FAULT define; release kernels never compile it.
enum HandoffFault : uint32_t {
    kFaultNone = 0,
    kFaultDoubleCredit = 1,  // the receiver credits chunk 0 twice               -> C1/C2 on its owner, or C4/C9 on
                             //                                                     the other receiver (early send)
    kFaultWrongCanary = 2,   // the producer writes c + 1 as chunk c's canary    -> C8 on the receiver
    kFaultShortPush = 3,     // the receiver pushes chunk 1's nkd one tile short -> C3 at the slot's next push
    kFaultWrongOwner = 4,    // the receiver credits chunk 1 to producer 0       -> C9 on both sides
    kFaultNoCredit = 5,      // the receiver never credits chunk 3               -> C9 on both sides
    kFaultCount = 6,
};

// Protocol tag: the LAST compile-time arg of the fused writer and the fused receiver reader. Both kernels
// static_assert on it, so an added, removed or reordered trailing compile-time arg on either side fails to compile.
constexpr uint32_t kHandoffTagVersion = 1;
constexpr uint32_t kHandoffTag = 0x47444E00u | kHandoffTagVersion;  // 'G' 'D' 'N' <version>

}  // namespace gdn_handoff
