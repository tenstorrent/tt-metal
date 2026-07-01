// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// transpose_rm_common.h — shared CB layout for transpose_rm.
//
// transpose_rm swaps the last two dims of a ROW_MAJOR fp32/bf16 tensor of
// shape (B, A, C) → (B, C, A) by working on 32×32 sub-blocks ("tiles").
// The reader gathers one source block into L1 then does an in-place
// element swap; the writer scatters that block to the transposed
// destination address.  Constraint: A, C ≥ 32 and multiples of 32.
//
// Block size (T_BLOCK = 32) is a compile-time invariant of the kernel —
// it matches the Tensix tile side, but the kernel does NOT use SFPU /
// matmul / packer; it's pure NoC + scalar L1 swaps.  Holding the same
// constant on both sides of the kernel set keeps math simple.

#pragma once

#include <cstdint>

constexpr uint32_t T_BLOCK         = 32u;
constexpr uint32_t T_BLOCK_ELEMS   = T_BLOCK * T_BLOCK;       // 1024
constexpr uint32_t T_BLOCK_BYTES_FP32 = T_BLOCK_ELEMS * 4u;   // 4096
constexpr uint32_t T_BLOCK_BYTES_BF16 = T_BLOCK_ELEMS * 2u;   // 2048

// Single CB used for the source block staging area.  Double-buffered
// (2 slots) to let the reader stay one block ahead of the writer.
constexpr uint32_t CB_TR_BLOCK = 0;
