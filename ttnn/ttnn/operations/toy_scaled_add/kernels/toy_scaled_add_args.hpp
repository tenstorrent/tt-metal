// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers and runtime-argument slots of the toy_scaled_add kernels. The kernels and the
// host that launches them index every slot through these names, so the two sides agree by
// construction.
//
// The arguments are split by how often they change:
//   * common runtime args carry what changes from call to call (buffer addresses, alpha). A kernel
//     has one copy for all of its cores, so refreshing them costs the same on one core as on the
//     whole grid;
//   * per-core runtime args carry the work split, which the tensor shapes fix.
//
// Compile-time args: the named arg "Wt" (tiles per row) on every kernel; on the readers, the tensor
// accessor args of a, b and gamma in that order (the sharded reader: gamma only). gamma's accessor args
// are always present, a placeholder when there is no gamma, so the offsets of everything after them
// never depend on whether gamma was given.
namespace toy_scaled_add {

// Dense from 0: every launch sends one circular-buffer config slot per index up to the highest one
// the program uses. gamma, the only optional buffer, takes the last id, so a program without it
// leaves no hole.
namespace cb {
constexpr uint32_t A = 0;
constexpr uint32_t B = 1;
constexpr uint32_t OUT = 2;
constexpr uint32_t GAMMA = 3;
}  // namespace cb

// Per-core runtime args, the same layout on every kernel: the block of tile-rows this core owns.
namespace core_arg {
enum : uint32_t { ROW_START, NUM_ROWS, COUNT };
}  // namespace core_arg

// Common runtime args of the interleaved reader. GAMMA_ADDR is 0 when there is no gamma.
namespace reader_arg {
enum : uint32_t { A_ADDR, B_ADDR, GAMMA_ADDR, COUNT };
}  // namespace reader_arg

// Common runtime args of the sharded reader: a and b are already in this core's L1.
namespace sharded_reader_arg {
enum : uint32_t { GAMMA_ADDR, COUNT };
}  // namespace sharded_reader_arg

// Common runtime args of the interleaved writer.
namespace writer_arg {
enum : uint32_t { OUT_ADDR, COUNT };
}  // namespace writer_arg

// Common runtime args of the compute kernel. alpha travels as the bit pattern of an fp32.
namespace compute_arg {
enum : uint32_t { ALPHA_BITS, COUNT };
}  // namespace compute_arg

}  // namespace toy_scaled_add
