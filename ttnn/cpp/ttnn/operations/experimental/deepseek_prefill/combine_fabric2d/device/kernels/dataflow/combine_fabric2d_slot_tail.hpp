// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Wire format of a ring slot's metadata tail: the reader fills it, the producer consumes it. Indices are
// uint64_t words from (slot_base + token_size_bytes); the stride itself is CMBF2D_SLOT_TAIL_BYTES in the
// program factory, which also sizes the forwarding-buffer page to match.
constexpr uint32_t TAIL_FINAL_ADDR = 0;  // destination DRAM address on the FINAL destination chip
constexpr uint32_t TAIL_DST_CHIP = 1;    // final destination chip id; SENTINEL_DST_CHIP marks a sentinel
constexpr uint32_t TAIL_CMD = 2;
constexpr uint32_t TAIL_THIS_ADDR = 3;  // the address THIS hop writes to

constexpr uint64_t CMD_END = 0;  // end of stream; the slot carries no token
constexpr uint64_t CMD_FINAL_WRITE = 1;
constexpr uint64_t CMD_FORWARD = 2;

// A sentinel carries no usable token; it marks the end of a forwarding chunk. UINT64_MAX can never collide
// with a real chip id.
constexpr uint64_t SENTINEL_DST_CHIP = UINT64_MAX;
