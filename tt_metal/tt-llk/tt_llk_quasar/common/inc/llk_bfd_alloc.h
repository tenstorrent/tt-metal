// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_id.h"

// Buffer descriptor (BFD) id allocator.
//
// On Quasar a buffer descriptor is an L1 "view" (base address, tile dims, data format) consumed
// by the UNPACR/PACR instruction streams. BFD ids are an LLK-internal resource: nothing above the
// LLK API layer may hold or pass one. The 32-entry table is physically partitioned per TRISC, so
// threads never write each other's entries:
//
//   TRISC0 (unpack)       -> ids [0, 16)
//   TRISC1 (math)         -> none (math issues no TDMA; DFB fifo state does not exist on math)
//   TRISC2 (pack)         -> ids [16, 24)
//   TRISC3 (isolate-SFPU) -> ids [24, 32)
//
// Note on math kernels that take an operand/DFB id (e.g. llk_math_transpose_dest_init,
// llk_math_eltwise_unary_datacopy_init): the id is used only for shape/format lookup
// (get_operand_dst_format / get_operand_num_faces / ...) to configure the ALU — no math kernel
// touches a BFD, so the decoupling of BFD ids from DFB ids does not affect them. (Metal 2.0
// named accessors would eventually replace those id-based format lookups; that is orthogonal
// to the BFD work.)
//
// Each thread bump-allocates within its own partition and wraps to the partition base once
// exhausted (wrap-around is by design). An op's init call allocates an id, programs the table
// entry, and bakes the id into the MOP; execute calls reference the id implicitly through the
// MOP until the next init on the same resource.
//
// Wrap hazard contract: an entry handed out more than `partition size` allocations ago may be
// overwritten, so an op must be re-initialised (not just re-executed) once newer inits may have
// lapped its id. Metal's existing discipline (op re-init before re-execute, and CB wait/pop plus
// dest semaphore handoffs draining TDMA between ops) satisfies this; `generation` is a debug
// breadcrumb for spotting lapses in waveforms/dumps.
//
// This header is compute-TRISC only: it keys off COMPILE_FOR_TRISC (via ckernel_trisc_id.h) and
// must not be included from data-movement translation units.

namespace ckernel::trisc
{

// TDMA resources that consume buffer descriptors. One "current id" slot each; which slots a
// thread actually uses depends on its TRISC role (UNP_* on T0, PACK* on T2/T3).
enum class BfdResource : std::uint8_t
{
    UnpA = 0,
    UnpB,
    UnpS,
    UnpDest,
    Pack0,
    Pack1,
    Count
};

constexpr std::uint8_t BFD_TABLE_NUM_ENTRIES = 32;

// TRISC1 (math) gets no partition. Base/size are constexpr so allocation has no runtime table.
constexpr std::uint8_t bfd_partition_base(const std::uint32_t trisc)
{
    switch (trisc)
    {
        case 0:
            return 0; // unpack
        case 2:
            return 16; // pack
        case 3:
            return 24; // isolate-SFPU
        default:
            return 0; // math: unused (size 0)
    }
}

constexpr std::uint8_t bfd_partition_size(const std::uint32_t trisc)
{
    switch (trisc)
    {
        case 0:
            return 16; // unpack
        case 2:
            return 8; // pack
        case 3:
            return 8; // isolate-SFPU
        default:
            return 0; // math
    }
}

struct BfdAllocatorState
{
    std::uint8_t next;                                                   // next id to hand out; valid only when initialized
    std::uint8_t current[static_cast<std::uint8_t>(BfdResource::Count)]; // most recent id per resource
    std::uint32_t generation;                                            // wrap count (debug breadcrumb)
    bool initialized;                                                    // lazy init: globals are bss zero-init only, no dynamic init on device
};

// One instance per per-TRISC binary (each TRISC compiles its own image with COMPILE_FOR_TRISC).
inline BfdAllocatorState bfd_state; // zero-init; next initialized lazily to the partition base

/**
 * @brief Allocate the next buffer descriptor id for this thread's partition and record it as the
 * current id for resource R. Wraps to the partition base when the partition is exhausted.
 * @tparam R: TDMA resource the id will drive (sets the bfd_current<R>() slot)
 * @return buffer descriptor id to program and bake into the MOP
 */
template <BfdResource R>
inline std::uint8_t bfd_alloc()
{
    static_assert(ckernel::TRISC_ID != 1, "math TRISC owns no buffer descriptors");
    static_assert(R < BfdResource::Count, "invalid BFD resource");
    constexpr std::uint8_t base = bfd_partition_base(ckernel::TRISC_ID);
    constexpr std::uint8_t end  = base + bfd_partition_size(ckernel::TRISC_ID);

    if (!bfd_state.initialized)
    {
        bfd_state.next        = base;
        bfd_state.initialized = true;
    }

    const std::uint8_t id                           = bfd_state.next;
    bfd_state.current[static_cast<std::uint8_t>(R)] = id;
    const std::uint8_t next                         = id + 1;
    bfd_state.next                                  = (next >= end) ? base : next;
    bfd_state.generation += (next >= end) ? 1 : 0;
    return id;
}

/**
 * @brief Most recently allocated id for resource R on this thread.
 * @note Only meaningful after at least one bfd_alloc<R>() on this thread; before that it is 0.
 */
template <BfdResource R>
inline std::uint8_t bfd_current()
{
    return bfd_state.current[static_cast<std::uint8_t>(R)];
}

} // namespace ckernel::trisc
