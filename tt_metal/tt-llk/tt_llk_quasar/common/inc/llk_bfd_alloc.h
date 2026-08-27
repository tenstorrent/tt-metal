// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "ckernel_trisc_id.h"
#include "llk_assert.h"

// Buffer descriptor (BFD) id allocator.
//
// On Quasar a buffer descriptor is an L1 "view" (base address, tile dims, data format) consumed
// by the UNPACR/PACR instruction streams. BFD ids are an LLK-internal resource: nothing above the
// LLK API layer may hold or pass one. The 32-entry table is physically partitioned per TRISC, so
// threads never write each other's entries:
//
//   TRISC0 (unpack)       -> ids [0, 16)
//   TRISC1 (math)         -> none (math issues no TDMA)
//   TRISC2 (pack)         -> ids [16, 24)
//   TRISC3 (isolate-SFPU) -> ids [24, 32)
//
// Note on math kernels that take an operand id (e.g. llk_math_transpose_dest_init,
// llk_math_eltwise_unary_datacopy_init): the id is used only for shape/format lookup
// (get_operand_dst_format / get_operand_num_faces / ...) to configure the ALU — no math kernel
// touches a BFD, so this allocator does not affect them. (Metal 2.0 named accessors would
// eventually replace those id-based format lookups; that is orthogonal to the BFD work.)
//
// Each thread bump-allocates within its own partition and wraps to the partition base once
// exhausted (wrap-around is by design). An op's init call allocates an id, programs the table
// entry, and bakes the id into the MOP; execute calls reference the id implicitly through the
// MOP until the next init on the same resource.
//
// Wrap hazard contract: an entry handed out more than `partition size` allocations ago may be
// overwritten, so an op must be re-initialised (not just re-executed) once newer inits may have
// lapped its id. Metal's existing discipline (op re-init before re-execute, and CB wait/pop plus
// dest semaphore handoffs draining TDMA between ops) satisfies this.
//
// This header is compute-TRISC only: it keys off COMPILE_FOR_TRISC (via ckernel_trisc_id.h) and
// must not be included from data-movement translation units.

namespace ckernel::trisc
{

// Real UNPACR/PACR hardware engines that consume buffer descriptors. One "current id" slot each;
// compile-time ownership ties each engine to a TRISC role (see bfd_engine_owned_by_trisc).
enum class BfdResource : std::uint8_t
{
    Unp0 = 0,
    Unp1,
    Pack0,
    Pack1,
    Count
};

// Sentinel for "no id allocated yet" in current[]. Real ids are 0..31, so 128 is safely out of
// range and fits uint8_t. current[] is bss zero-init on device (0 is a valid id), so it must be
// set to this sentinel inside the lazy-init block, not via a static initializer.
constexpr std::uint8_t BFD_ID_INVALID = 128;

constexpr bool bfd_engine_owned_by_trisc(const BfdResource engine, const std::uint32_t trisc)
{
    switch (engine)
    {
        case BfdResource::Unp0:
        case BfdResource::Unp1:
            return trisc == 0;
        case BfdResource::Pack0:
            return trisc == 2;
        case BfdResource::Pack1:
            return trisc == 3;
        default:
            return false;
    }
}

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
    std::uint8_t current[static_cast<std::uint8_t>(BfdResource::Count)]; // most recent id per engine; BFD_ID_INVALID until first alloc
    bool initialized;                                                    // lazy init: globals are bss zero-init only, no dynamic init on device
};

// One instance per hardware TRISC thread (one per Neo and TRISC role). thread_local gives each Neo's
// TRISC its own allocator, matching the per-TRISC partitioning of each Neo's BFD table. A
// single shared instance races the id allocator two ways when num_threads > 1 (see tt-llk#1678):
// an unfenced read-modify-write on next hands the same id to two threads, and the unfenced lazy
// init lets one role hand out ids from another role's partition. Mirrors trisc::dest_register_offset:
// ENV_LLK_INFRA (standalone LLK infra, no firmware TU) uses a plain static; the metal build
// declares it extern thread_local and defines it in firmware (tt_metal/hw/firmware/src/tt-2xx/trisc.cc).
#ifdef ENV_LLK_INFRA
static BfdAllocatorState bfd_state; // zero-init; next initialized lazily to the partition base
#else
extern thread_local BfdAllocatorState bfd_state; // defined in tt_metal/hw/firmware/src/tt-2xx/trisc.cc
#endif

/**
 * @brief Allocate the next buffer descriptor id for this thread's partition and record it as the
 * current id for engine E. Wraps to the partition base when the partition is exhausted.
 * @tparam E: hardware engine the id will drive (sets the bfd_current<E>() slot)
 * @return buffer descriptor id to program and bake into the MOP
 */
template <BfdResource E>
inline std::uint8_t bfd_alloc()
{
    static_assert(ckernel::TRISC_ID != 1, "math TRISC owns no buffer descriptors");
    static_assert(E < BfdResource::Count, "invalid BFD engine");
    static_assert(bfd_engine_owned_by_trisc(E, ckernel::TRISC_ID), "BFD engine not owned by compiling TRISC");
    constexpr std::uint8_t base = bfd_partition_base(ckernel::TRISC_ID);
    constexpr std::uint8_t end  = base + bfd_partition_size(ckernel::TRISC_ID);

    if (!bfd_state.initialized)
    {
        bfd_state.next = base;
        for (std::uint8_t i = 0; i < static_cast<std::uint8_t>(BfdResource::Count); ++i)
        {
            bfd_state.current[i] = BFD_ID_INVALID;
        }
        bfd_state.initialized = true;
    }

    const std::uint8_t id                           = bfd_state.next;
    bfd_state.current[static_cast<std::uint8_t>(E)] = id;
    const std::uint8_t next                         = id + 1;
    bfd_state.next                                  = (next >= end) ? base : next;
    return id;
}

/**
 * @brief Most recently allocated id for engine E on this thread.
 * @note Asserts if no bfd_alloc<E>() has run yet on this thread.
 */
template <BfdResource E>
inline std::uint8_t bfd_current()
{
    static_assert(ckernel::TRISC_ID != 1, "math TRISC owns no buffer descriptors");
    static_assert(E < BfdResource::Count, "invalid BFD engine");
    static_assert(bfd_engine_owned_by_trisc(E, ckernel::TRISC_ID), "BFD engine not owned by compiling TRISC");
    // current[] is bss zero before the first bfd_alloc (0 is a valid id), and the BFD_ID_INVALID
    // sentinel is only written inside bfd_alloc's lazy-init; gate on initialized so a bfd_current
    // that races ahead of any allocation trips the assert instead of returning a bogus id 0.
    LLK_ASSERT(bfd_state.initialized && bfd_state.current[static_cast<std::uint8_t>(E)] != BFD_ID_INVALID, "bfd_current before first bfd_alloc");
    return bfd_state.current[static_cast<std::uint8_t>(E)];
}

/**
 * @brief One-stop op-init sequence: allocate the next buffer descriptor id for engine E,
 * construct the descriptor from the L1 buffer info, and program the table entry.
 * @tparam E: hardware engine the id will drive
 * @tparam MODE: L1 access mode (Continuous, or Strided for PACR/UNPACR_STRIDE tiny-tiles)
 * @param tensor_shape: Tile/face dimensions and shape of the buffer
 * @param l1_base_addr: base address of the buffer in L1 (16B units)
 * @param l1_data_format: L1 data encoding format
 */
template <BfdResource E, L1AccessMode MODE = L1AccessMode::Continuous>
inline void bfd_alloc_and_program(const TensorShape& tensor_shape, const std::uint32_t l1_base_addr, const std::uint32_t l1_data_format)
{
    const std::uint8_t id = bfd_alloc<E>();
    _configure_buf_desc_table_(id, construct_buf_desc<MODE>(tensor_shape, l1_base_addr, l1_data_format));
}

} // namespace ckernel::trisc
