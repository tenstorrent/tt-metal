// SPDX-License-Identifier: Apache-2.0
// Scratch-branch toggle header for the SDPA zone decomposition campaign (handoff/revamp).
// Rewritten between runs by the campaign script; every kernel is JIT-recompiled per process.
#pragma once
#define SDPA_ZONES 0            // 1: accumulate + per-q-chunk raw zones compiled in (PROFILE_KERNEL build only)
#define SDPA_ABL_READER_STUB 0  // A4: reader reserves/pushes K and V CBs without NoC reads
#define SDPA_ABL_MASK_OFF 0     // A2: causal lightweight mask bracket skipped on every k chunk
#define SDPA_ABL_EXP_STUB 0     // A6: softmax exp_packthread_tile calls removed (STALLWAIT kept)
#define SDPA_ABL_BARRIER_THR 0  // A4b: >0 overrides the reader barrier_threshold (reads in flight per barrier)
