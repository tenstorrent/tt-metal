// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "internal/local_mem/address_access.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#if !defined(ARCH_QUASAR)
// ckernel::load_blocking (WH/BH store-drain). Unusable from a Quasar DM build: ckernel.h ->
// ckernel_addrmod.h -> ckernel_trisc_id.h #errors unless COMPILE_FOR_TRISC, and Quasar's ckernel.h has
// no load_blocking. Same guard rationale as common.hpp. The Quasar fence below uses flush_l2_cache_range.
#include "ckernel.h"
#endif
#include "experimental/kernel_args.h"

inline __attribute__((always_inline)) void fill_pad_dfb_with_val(
    Scratchpad<uint32_t>& pad, const uint32_t num_bytes, const uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = &pad[0];

    // Round up so a non-4-byte-aligned tail stick is fully filled (the loop-back read consumes all num_bytes).
    const uint32_t num_words = (num_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; ++i) {
        ptr[i] = val;
    }
}

template <typename StreamState>
inline __attribute__((always_inline)) void read_input_stick_into_l1(
    Noc& noc,
    const StreamState& s,
    uint32_t& i_page,
    CoreLocalMem<uint32_t> dst,
    const uint32_t num_input_pages_in_row,
    const uint32_t stick_size_bytes) {
    if (num_input_pages_in_row == 1) {
        // Width fits in a single page: index the accessor with the flat page id directly.
        // `noc_async_read_sharded` derives pages-per-row from the (rank-squeezed) dspec shape,
        // which is wrong when an outer dim is sharded and the width is a single page (the
        // width-page dim gets squeezed away and an inner dim is mistaken for the row width).
        noc.async_read(s, dst, stick_size_bytes, {.page_id = i_page, .offset_bytes = 0}, {});
    } else {
        const uint32_t stick_id = i_page / num_input_pages_in_row;
        tt::data_movement::common::noc_async_read_sharded(
            noc, dst, s, stick_id, /*offset=*/0, /*size=*/stick_size_bytes);
    }
    i_page += num_input_pages_in_row;
}

void kernel_main() {
    auto num_sticks_per_core = get_arg(args::num_sticks_per_core);
    auto num_sticks_per_barrier = get_arg(args::num_sticks_per_barrier);
    auto start_page_id = get_arg(args::start_page_id);
    auto front_pad_n = get_arg(args::front_pad_n);
    auto front_pad_c = get_arg(args::front_pad_c);
    auto front_pad_h = get_arg(args::front_pad_h);

    constexpr auto N = get_arg(args::N);
    constexpr auto H = get_arg(args::H);
    constexpr auto C = get_arg(args::C);
    constexpr auto stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr auto N_padded = get_arg(args::N_padded);
    constexpr auto H_padded = get_arg(args::H_padded);
    constexpr auto C_padded = get_arg(args::C_padded);
    constexpr auto stick_size_padded = get_arg(args::stick_size_padded);
    constexpr auto stick_size_padded_front = get_arg(args::stick_size_padded_front);
    constexpr auto stick_size_padded_aligned = get_arg(args::stick_size_padded_aligned);

    constexpr bool not_pad_by_zero = get_arg(args::not_pad_by_zero) == 1;
    constexpr uint32_t front_padding = stick_size_padded_front;
    constexpr bool unaligned = get_arg(args::unaligned) == 1;

    constexpr auto num_input_pages_in_row = get_arg(args::num_input_pages_in_row);

    uint32_t packed_pad_value = 0;
    if constexpr (not_pad_by_zero) {
        packed_pad_value = get_arg(args::packed_pad_value);
    }

    DataflowBuffer dfb_in0_exp(dfb::in0);
    Scratchpad<uint32_t> pad(scratch::pad);
    // The realignment staging buffer is bound only when the host allocated it (front padding, or
    // an unaligned padded stick). A kernel may not name a DFB it has not bound, and `if constexpr`
    // does not suppress that name lookup, so every reference to it is gated at the preprocessor.
#ifdef PAD_ALIGN_DFB
    Scratchpad<uint32_t> pad_align(scratch::pad_align);
#endif

    const auto s = TensorAccessor(tensor::src);
    Noc noc;

    fill_pad_dfb_with_val(pad, stick_size_padded, packed_pad_value);
    // The fill above is baby-RISCV CPU stores; the per-stick loop below loop-back noc.async_read's the pad
    // scratchpad as its source. The stores must be made visible to the NoC before that first read, but the
    // mechanism differs by arch:
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    // Quasar DM: CPU stores land in the RISC's L1D/L2, and the NoC sources TL1 directly. A local load does
    // NOT publish to TL1 here (and ckernel::load_blocking is unavailable on Quasar), so flush the filled
    // range so the reads see the fill. No-op on WH/BH. Matches fill_rm_interleaved.cpp / common.hpp (#51763).
    flush_l2_cache_range(
        static_cast<uintptr_t>(local_mem::detail::AddressAccess::address(pad.local_mem())),
        static_cast<size_t>(stick_size_padded));
#else
    // WH/BH: a baby-RISCV store can retire before its write-request lands in L1, and the RISCV core and NoC
    // are different L1 clients with no program-order guarantee (WormholeB0/TensixTile/BabyRISCV/
    // MemoryOrdering.md). load_blocking the last filled word (blocking load + memory clobber) forces the fill
    // to be processed before the first loop-back read. Index must match fill_pad_dfb_with_val's ceil-rounded
    // word count: floor(size/4)-1 would fence on the word before a non-4B tail (racing its store) and
    // underflow for a sub-4B stick. stick_size_padded > 0, so the ceil count is >= 1 and never underflows.
    constexpr uint32_t pad_last_word = (stick_size_padded + sizeof(uint32_t) - 1) / sizeof(uint32_t) - 1;
    volatile tt_l1_ptr uint32_t* pad_last_word_ptr = &pad.local_mem()[pad_last_word];
    (void)ckernel::load_blocking(pad_last_word_ptr);
#endif

    uint32_t i_page = start_page_id;
    uint32_t curr_c = get_arg(args::start_dim_offset_c), curr_h = get_arg(args::start_dim_offset_h),
             curr_n = get_arg(args::start_dim_offset_n);
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        dfb_in0_exp.reserve_back(num_sticks_per_barrier);
        uint32_t l1_write_addr = dfb_in0_exp.get_write_ptr();

        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            bool read_stick = (curr_h >= front_pad_h and curr_h < H) and (curr_c >= front_pad_c and curr_c < C) and
                              (curr_n >= front_pad_n and curr_n < N);
            {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                // Loop-back read of this core's pad scratchpad.
                noc.async_read(pad, dst, stick_size_padded, {.offset_bytes = 0}, {.offset_bytes = 0});
                noc.async_read_barrier();
            }
            if (read_stick) {
#ifdef PAD_ALIGN_DFB
                if constexpr (front_padding) {
                    read_input_stick_into_l1(
                        noc, s, i_page, pad_align.local_mem(), num_input_pages_in_row, stick_size_bytes);
                    noc.async_read_barrier();
                    // Relocate the staged stick with the shared CPU copy (common.hpp): on Quasar DM it reads
                    // the source through the uncached L1 alias (the NoC just wrote it; a stale D$ line could
                    // otherwise survive), writes the destination through the uncached alias and fences the
                    // stores before the DFB credit is posted. A raw memmove into the uncached write pointer
                    // would let the consumer observe the push_back before the bytes land. On WH/BH it is the
                    // usual memmove + load_blocking drain, and it casts through uintptr_t (the bare
                    // (void*)(uint32_t) cast is -Werror=int-to-pointer-cast on Quasar's rv64 DM RISCs).
                    tt::data_movement::common::copy_via_memmove<false>(
                        CoreLocalMem<uint32_t>(l1_write_addr + stick_size_padded_front),
                        pad_align.local_mem(),
                        stick_size_bytes);
                } else if constexpr (unaligned) {
                    read_input_stick_into_l1(
                        noc, s, i_page, pad_align.local_mem(), num_input_pages_in_row, stick_size_bytes);
                    noc.async_read_barrier();
                    CoreLocalMem<uint32_t> dst(l1_write_addr);
                    // Loop-back read of this core's realignment staging scratchpad.
                    noc.async_read(pad_align, dst, stick_size_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
                } else
#endif
                {
                    read_input_stick_into_l1(
                        noc,
                        s,
                        i_page,
                        CoreLocalMem<uint32_t>(l1_write_addr),
                        num_input_pages_in_row,
                        stick_size_bytes);
                }
            }
            l1_write_addr += stick_size_padded_aligned;
            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }
        noc.async_read_barrier();
        dfb_in0_exp.push_back(num_sticks_per_barrier);
    }
}
