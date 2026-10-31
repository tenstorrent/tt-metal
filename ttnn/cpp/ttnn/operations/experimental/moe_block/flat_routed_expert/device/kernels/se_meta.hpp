// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Layout of the dynamic-counts meta page (CB 6) a data-movement kernel hands its compute kernel (se_dyn.hpp,
// se_dyn_publish), in uint32 words:
//   [0] n_act (schedule entries), [1] num_v (sub-blocks over all entries), [SE_META_SUBS + a] entry a's sub-blocks,
//   [SE_META_SMALL] small-M role split flag, [SE_META_GU + a] entry a's gate/up weight use: load index (bits 0-7),
//   ring region of that load (8-15), last use of the load (bit 16), [SE_META_LMT + a] row tiles holding tokens in
//   entry a's last sub-block (the compute skips the rest).
#pragma once
#include <stdint.h>

#ifndef SE_MAX_E
#define SE_MAX_E 16
#endif
constexpr uint32_t SE_MAX_V = 2 * SE_MAX_E;  // schedule entries: a pinned expert's chunks + the others
static_assert(SE_MAX_V < 255, "SeDyn keeps entry / load indices in 8 bits (0xFF: none)");
constexpr uint32_t SE_META_SUBS = 2;
constexpr uint32_t SE_META_SMALL = SE_META_SUBS + SE_MAX_V;
constexpr uint32_t SE_META_GU = SE_META_SMALL + 1;
constexpr uint32_t SE_META_LMT = SE_META_GU + SE_MAX_V;
constexpr uint32_t SE_META_WORDS = SE_META_LMT + SE_MAX_V;  // the page must hold this many words (host: CB 6 page)
