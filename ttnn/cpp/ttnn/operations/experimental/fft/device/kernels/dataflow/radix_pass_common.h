// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// radix_pass_common.h — Shared CB layout for the fused FFT + post-twiddle
// device op (ttnn::prim::fft_radix_pass).  Reuses the entire BATCH_FFT
// CB layout (0..20) and adds two more CBs for the post-twiddle staging
// tile, when APPLY_POST_TWIDDLE=1.
//
// CB_PT_R / CB_PT_I are 1-slot scratch buffers used only by
// radix_pass_reader.cpp — produced AND consumed inside the same kernel,
// solely as an L1 landing pad for the noc_async_read_tile of the
// twiddle table.  No other kernel touches them.

#pragma once

#include "batch_fft_common.h"

constexpr uint32_t CB_PT_R = 21;
constexpr uint32_t CB_PT_I = 22;

// Total CB count for the radix-pass variants.  Includes BATCH_NUM_CBS
// (fp32 path, 0..16) + CB_PT_R/I.  When INPUT_BF16=1, the bf16 staging
// CBs (17..20) are also allocated, bringing the total to RADIX_NUM_CBS_BF16.
constexpr uint32_t RADIX_NUM_CBS      = 19;   // 0..16 + 21..22
constexpr uint32_t RADIX_NUM_CBS_BF16 = 23;   // 0..20 + 21..22
