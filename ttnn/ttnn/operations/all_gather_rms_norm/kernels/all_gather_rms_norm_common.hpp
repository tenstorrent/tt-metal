// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tools/profiler/kernel_profiler.hpp"

// Opt-in per-stage zones (host env AGRMS_ZONES=1 -> define AGRMS_ZONES); compiled out otherwise.
#ifdef AGRMS_ZONES
#define AGZ(name) DeviceZoneScopedN(name)
#else
#define AGZ(name)
#endif

// CB indices — must match all_gather_rms_norm_program_descriptor.py.
namespace agrms_cb {
constexpr uint32_t cb_x = 0;  // landing slot + resident member block (FIRST CB: same address on every core / chip)
constexpr uint32_t cb_stage = 1;
constexpr uint32_t cb_gamma = 2;
constexpr uint32_t cb_beta = 3;
constexpr uint32_t cb_ones = 4;
constexpr uint32_t cb_mask = 5;
constexpr uint32_t cb_moment_a = 6;
constexpr uint32_t cb_moment_b = 7;
constexpr uint32_t cb_km = 8;
constexpr uint32_t cb_partial = 9;
constexpr uint32_t cb_gathered = 10;
constexpr uint32_t cb_rowstat = 11;
constexpr uint32_t cb_out = 12;
constexpr uint32_t cb_mstat = 13;
constexpr uint32_t cb_shift = 14;
constexpr uint32_t cb_kg = 15;
constexpr uint32_t cb_mu = 16;
constexpr uint32_t cb_affine_stage = 17;
constexpr uint32_t cb_norm = 18;
constexpr uint32_t cb_shift_mats = 19;
constexpr uint32_t cb_foreign = 20;
constexpr uint32_t cb_contrib = 21;
constexpr uint32_t cb_cent = 22;
}  // namespace agrms_cb
