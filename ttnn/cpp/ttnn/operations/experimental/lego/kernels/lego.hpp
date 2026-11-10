// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Lego blocks: the kernel side.
//
// A block is a logical unit of an existing op (a matmul's in0 reader, a reduce-scatter's writer, ...) packaged as
//
//     template <typename A> struct Block {
//         void init();      // read args, build accessors and connections
//         void run();       // the work
//         void teardown();  // barriers, connection close, semaphore resets
//     };
//
// The host composer (ttnn/ttnn/lego) generates one kernel per (core group, RISC) that declares the blocks placed
// there, then calls every init(), every run() and every teardown() in placement order. `A` is a generated struct of
// static constexpr members: the block's compile-time values plus the offsets of its slices of the kernel's runtime
// args (rt_base), common runtime args (crt_base) and positional compile-time args (cta_base, for TensorAccessorArgs).
// Blocks therefore never index args by absolute position; they read their own slice through the helpers below.

#pragma once

#include <cstdint>

// LEGO_ZONE(name): a device-profiler zone inside a block, compiled in only when the composer builds with zones
// (LEGO_ZONES=1 in the host environment).
#ifdef LEGO_ZONES
#include "api/debug/kernel_profiler.hpp"
#define LEGO_ZONE(name) DeviceZoneScopedN(name)
#else
#define LEGO_ZONE(name)
#endif

namespace lego {

// i-th per-core runtime arg of block A.
template <typename A>
FORCE_INLINE uint32_t rt(uint32_t i) {
    return get_arg_val<uint32_t>(A::rt_base + i);
}

// i-th common runtime arg of block A.
template <typename A>
FORCE_INLINE uint32_t crt(uint32_t i) {
    return get_common_arg_val<uint32_t>(A::crt_base + i);
}

// Absolute index of block A's i-th runtime arg (for APIs that take an arg index, e.g. fabric build_from_args).
template <typename A>
constexpr uint32_t rt_index(uint32_t i) {
    return A::rt_base + i;
}

}  // namespace lego
