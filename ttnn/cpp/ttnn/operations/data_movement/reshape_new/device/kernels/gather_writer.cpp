// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather writer: assembles each output unit staged by the gather reader (layout in gather_common.hpp) and writes
// its pages to the output. A padded unit is first covered with the padding pattern when fill_padding is set. A
// staged run that shares its offset within a 16-byte NOC word with its placement is moved by a local NOC read; any
// other run is copied by the CPU (one store drain per entry). Units alternate between two assembly slots, and
// entries between the two stage buffers, starting with the first.
//
// Compile-time args: unit_slot_bytes, page_slot_bytes, fill_padding, pad_bits
// Runtime args:      unit_begin, unit_end
// Bindings:          dfb::stage0, dfb::stage1 (consumer), scratch::assembly, tensor::dst

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_new/device/kernels/gather_common.hpp"

namespace {

using namespace reshape_new;

constexpr uint32_t unit_slot_bytes = get_arg(args::unit_slot_bytes);
constexpr uint32_t page_slot_bytes = get_arg(args::page_slot_bytes);
constexpr bool fill_padding = get_arg(args::fill_padding) != 0;
constexpr uint32_t pad_bits = get_arg(args::pad_bits);
// The address the NOC uses for a local L1 address (DFB pointers are the uncached alias on Quasar DM).
FORCE_INLINE uint32_t noc_l1_address(uint32_t addr) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    if (addr >= MEM_L1_UNCACHED_BASE) {
        addr -= MEM_L1_UNCACHED_BASE;
    }
#endif
    return addr;
}

// Makes the CPU stores up to and including the byte at `last_byte` visible to the NOC.
FORCE_INLINE void drain_cpu_stores(uint32_t last_byte) {
#if defined(ARCH_QUASAR)
    // copy_via_memmove stores through the uncached alias and fences them.
    (void)last_byte;
#else
    (void)ckernel::load_blocking(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(last_byte & ~uint32_t{3})));
#endif
}

// Copies a run with CPU loads and stores; the caller drains them before the NOC reads `to`. The L1 cache was
// invalidated when the entry was taken, so the loads see the staged bytes.
FORCE_INLINE void cpu_copy(uint32_t to, uint32_t from, uint32_t bytes) {
#if defined(ARCH_QUASAR)
    tt::data_movement::common::copy_via_memmove<true>(to, from, bytes);
#else
    if (((to | from | bytes) & 3) == 0) {
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(to));
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(from));
        for (uint32_t i = 0; i < bytes / sizeof(uint32_t); ++i) {
            dst[i] = src[i];
        }
    } else if (((to | from | bytes) & 1) == 0) {
        volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(static_cast<uintptr_t>(to));
        volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(static_cast<uintptr_t>(from));
        for (uint32_t i = 0; i < bytes / sizeof(uint16_t); ++i) {
            dst[i] = src[i];
        }
    } else {
        tt::data_movement::common::copy_via_memmove<true>(to, from, bytes);
    }
#endif
}

// Writes the padding pattern with CPU stores and makes it visible to the NOC before returning.
FORCE_INLINE void fill_pattern(uint32_t addr) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    addr += MEM_L1_UNCACHED_BASE;
#endif
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(addr));
    constexpr uint32_t num_words = unit_slot_bytes / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; ++i) {
        words[i] = pad_bits;
    }
#if defined(ARCH_QUASAR)
    __asm__ __volatile__("fence" ::: "memory");
    (void)words[num_words - 1];
#else
    (void)ckernel::load_blocking(&words[num_words - 1]);
#endif
}

}  // namespace

void kernel_main() {
    const uint32_t unit_begin = get_arg(args::unit_begin);
    const uint32_t unit_end = get_arg(args::unit_end);

    DataflowBuffer stage0(dfb::stage0);
    DataflowBuffer stage1(dfb::stage1);
    Noc noc;
    const auto dst = TensorAccessor(tensor::dst);
    Scratchpad<uint8_t> assembly(scratch::assembly);
    const uint32_t slots = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(assembly.local_mem()));
    const uint32_t pattern = slots + 2 * unit_slot_bytes;
    if constexpr (fill_padding) {
        fill_pattern(pattern);
    }

    uint32_t stage_index = 0;
    for (uint32_t unit = unit_begin; unit < unit_end; ++unit) {
        const uint32_t slot = slots + ((unit - unit_begin) & 1) * unit_slot_bytes;
        uint32_t page = 0;
        uint32_t page_offset = 0;
        uint32_t bytes = 0;
        uint32_t pages = 0;
        bool assembled = false;
        uint32_t flags = 0;
        do {
            uint32_t entry;
            if (stage_index == 0) {
                stage0.wait_front(1);
                entry = stage0.get_read_ptr();
            } else {
                stage1.wait_front(1);
                entry = stage1.get_read_ptr();
            }
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* header =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(entry));
            flags = header[kHeaderFlags];
            const uint32_t count = header[kHeaderCount];
            if (flags & kFlagFirst) {
                page = header[kHeaderPage];
                page_offset = header[kHeaderPageOffset];
                bytes = header[kHeaderUnitBytes];
                pages = header[kHeaderPageCount];
                // The slot is reused every other unit: its last writes must have left L1.
                noc.async_writes_flushed();
                if constexpr (fill_padding) {
                    if (flags & kFlagPadded) {
                        noc.async_read(
                            UnicastEndpoint{},
                            CoreLocalMem<uint32_t>(slot),
                            pages * page_slot_bytes,
                            tt::data_movement::common::self_l1_src_args(noc, pattern),
                            {});
                        noc.async_read_barrier();
                        assembled = true;
                    }
                }
            }
            uint32_t last_cpu_byte = 0;
            bool cpu_copied = false;
            for (uint32_t r = 0; r < count; ++r) {
                volatile tt_l1_ptr uint32_t* record = header + kHeaderWords + kRecordWords * r;
                const uint32_t unit_offset = record[0];
                const uint32_t staged_offset = record[1];
                const uint32_t run_bytes = record[2];
                const uint32_t to = slot + unit_offset;
                const uint32_t from = entry + staged_offset;
                if (((to ^ from) & (kNocWordBytes - 1)) == 0) {
                    noc.async_read(
                        UnicastEndpoint{},
                        CoreLocalMem<uint32_t>(to),
                        run_bytes,
                        tt::data_movement::common::self_l1_src_args(noc, noc_l1_address(from)),
                        {});
                } else {
                    cpu_copy(to, from, run_bytes);
                    last_cpu_byte = to + run_bytes - 1;
                    cpu_copied = true;
                }
            }
            if (cpu_copied) {
                drain_cpu_stores(last_cpu_byte);
            }
            assembled = assembled || count != 0;
            noc.async_read_barrier();
            if (stage_index == 0) {
                stage0.pop_front(1);
            } else {
                stage1.pop_front(1);
            }
            stage_index ^= 1;
        } while (!(flags & kFlagLast));

        if (assembled) {
            for (uint32_t p = 0; p < pages; ++p) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(slot + p * page_slot_bytes),
                    dst,
                    bytes,
                    {},
                    {.page_id = page + p, .offset_bytes = page_offset});
            }
        }
    }
    noc.async_write_barrier();
}
