// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather writer: assembles each output unit staged by the gather reader (layout in gather_common.hpp) and writes it
// to the output. A padded unit is first covered with the padding pattern when fill_padding is set. A staged run that
// shares its offset within a 16-byte NOC word with its placement is moved by a local NOC read; any other run is
// copied by the CPU. Units alternate between two assembly slots.
//
// Compile-time args: unit_slot_bytes, fill_padding, pad_bits
// Runtime args:      unit_begin, unit_end
// Bindings:          dfb::stage (consumer), scratch::assembly, tensor::dst

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

    DataflowBuffer stage(dfb::stage);
    Noc noc;
    const auto dst = TensorAccessor(tensor::dst);
    Scratchpad<uint8_t> assembly(scratch::assembly);
    const uint32_t slots = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(assembly.local_mem()));
    const uint32_t pattern = slots + 2 * unit_slot_bytes;
    if constexpr (fill_padding) {
        fill_pattern(pattern);
    }

    for (uint32_t unit = unit_begin; unit < unit_end; ++unit) {
        const uint32_t slot = slots + ((unit - unit_begin) & 1) * unit_slot_bytes;
        uint32_t page = 0;
        uint32_t page_offset = 0;
        uint32_t bytes = 0;
        bool assembled = false;
        uint32_t flags = 0;
        do {
            stage.wait_front(1);
            invalidate_l1_cache();
            const uint32_t entry = stage.get_read_ptr();
            volatile tt_l1_ptr uint32_t* header =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(entry));
            flags = header[kHeaderFlags];
            const uint32_t count = header[kHeaderCount];
            if (flags & kFlagFirst) {
                page = header[kHeaderPage];
                page_offset = header[kHeaderPageOffset];
                bytes = header[kHeaderUnitBytes];
                // The slot is reused every other unit: its last write must have left L1.
                noc.async_writes_flushed();
                if constexpr (fill_padding) {
                    if (flags & kFlagPadded) {
                        noc.async_read(
                            UnicastEndpoint{},
                            CoreLocalMem<uint32_t>(slot),
                            bytes,
                            tt::data_movement::common::self_l1_src_args(noc, pattern),
                            {});
                        noc.async_read_barrier();
                        assembled = true;
                    }
                }
            }
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
                    tt::data_movement::common::copy_via_memmove<false>(to, from, run_bytes);
                }
            }
            assembled = assembled || count != 0;
            noc.async_read_barrier();
            stage.pop_front(1);
        } while (!(flags & kFlagLast));

        if (assembled) {
            noc.async_write(
                CoreLocalMem<uint32_t>(slot), dst, bytes, {}, {.page_id = page, .offset_bytes = page_offset});
        }
    }
    noc.async_write_barrier();
}
