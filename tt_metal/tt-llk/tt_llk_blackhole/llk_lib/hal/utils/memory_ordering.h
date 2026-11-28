// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace hal::detail
{

// TODO: #58443 Remove the duplicate ckernel::load_blocking and store_blocking.
// Ordering rules: tt-isa-documentation BlackholeA0/TensixTile/BabyRISCV/MemoryOrdering.md.

/**
 * @brief Load a 32-bit word and stall the RISC until the load completes.
 *
 * The AND of the loaded value into x0 needs the data, so the pipeline waits for the load. The memory clobber
 * keeps the compiler from moving later accesses above it.
 *
 * @param ptr: Address to read.
 * @return The loaded word.
 */
inline __attribute__((always_inline)) std::uint32_t load_blocking(volatile std::uint32_t* ptr)
{
    std::uint32_t raw;
    asm volatile(
        "lw %[raw], (%[ptr])\n\t"
        "and x0, x0, %[raw]"
        : [raw] "=r"(raw)
        : [ptr] "r"(ptr)
        : "memory");
    return raw;
}

/**
 * @brief Store a 32-bit word and stall the RISC until the store completes.
 *
 * A load from the same address completes after the store, and the AND of its value into x0 waits for the load.
 * The memory clobber keeps the compiler from moving later accesses above it.
 *
 * @param ptr: Address to write.
 * @param value: Word to store.
 */
inline __attribute__((always_inline)) void store_blocking(volatile std::uint32_t* ptr, std::uint32_t value)
{
    asm volatile(
        "sw %[raw], (%[ptr])\n\t"
        "lw %[raw], (%[ptr])\n\t"
        "and x0, x0, %[raw]"
        : [raw] "+r"(value)
        : [ptr] "r"(ptr)
        : "memory");
}

} // namespace hal::detail
