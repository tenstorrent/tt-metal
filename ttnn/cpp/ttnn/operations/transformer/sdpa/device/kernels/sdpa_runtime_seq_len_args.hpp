// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// With SDPA_RUNTIME_SEQ_LEN defined, the SDPA reader, writer and compute kernels read the values that
// depend on the Q and K sequence lengths from common runtime args instead of compile-time args, so a
// new sequence length reuses the compiled kernels. The program factory writes the same list to every
// kernel; each kernel reads the entries it uses.
namespace sdpa_seq_len_args {
constexpr std::uint32_t SKT = 0;           // K length in tiles, padded to the K chunk
constexpr std::uint32_t VALID_SQT = 1;     // unpadded Q length in tiles
constexpr std::uint32_t VALID_SKT = 2;     // unpadded K length in tiles
constexpr std::uint32_t Q_NUM_CHUNKS = 3;  // Q chunks per head
constexpr std::uint32_t K_NUM_CHUNKS = 4;  // K chunks per head
constexpr std::uint32_t SK = 5;            // unpadded K length in rows
constexpr std::uint32_t NUM_ARGS = 6;
}  // namespace sdpa_seq_len_args
