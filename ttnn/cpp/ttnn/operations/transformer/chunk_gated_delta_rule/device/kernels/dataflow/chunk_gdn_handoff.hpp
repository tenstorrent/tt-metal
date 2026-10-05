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

// Protocol tag: the LAST compile-time arg of the fused writer and the fused receiver reader. Both kernels
// static_assert on it, so an added, removed or reordered trailing compile-time arg on either side fails to compile.
constexpr uint32_t kHandoffTagVersion = 1;
constexpr uint32_t kHandoffTag = 0x47444E00u | kHandoffTagVersion;  // 'G' 'D' 'N' <version>

}  // namespace gdn_handoff
