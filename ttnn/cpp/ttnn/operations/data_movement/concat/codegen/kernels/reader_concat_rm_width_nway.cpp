// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Width concat of three or more row-major inputs: assembles each output stick from all of its input sticks.
// A stick that fills its page and lands on a NOC-aligned output offset is read directly into the output CB; any
// other is staged in scratch and copied in by RISC stores.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ckernel.h"

constexpr uint32_t cb_out = get_compile_time_arg_val(0);
constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
constexpr uint32_t N_INPUTS = get_compile_time_arg_val(2);
constexpr uint32_t OUT_PAGE_SIZE = get_compile_time_arg_val(3);
constexpr uint32_t BATCH = get_compile_time_arg_val(4);
constexpr uint32_t SRC_NOC_ALIGNMENT = get_compile_time_arg_val(5);
constexpr uint32_t FIRST_ACCESSOR_ARGS = 6;

constexpr uint32_t base_rt = 0;
constexpr uint32_t stick_rt = base_rt + N_INPUTS;
constexpr uint32_t page_rt = stick_rt + N_INPUTS;

constexpr auto source_args = make_tensor_accessor_args_tuple<N_INPUTS, FIRST_ACCESSOR_ARGS>();

// Each input carries its own accessor type (DRAM vs L1), so the one to read through is picked by walking the inputs.
template <uint32_t Index>
FORCE_INLINE void read_source_page(
    Noc& noc, CircularBuffer& dst_cb, uint32_t input, uint32_t page_id, uint32_t dst_offset) {
    if constexpr (Index < N_INPUTS) {
        if (input == Index) {
            constexpr auto args = std::get<Index>(source_args);
            const uint32_t page_size = get_common_arg_val<uint32_t>(page_rt + Index);
            const auto accessor = TensorAccessor(args, get_common_arg_val<uint32_t>(base_rt + Index), page_size);
            noc.async_read(accessor, dst_cb, page_size, {.page_id = page_id}, {.offset_bytes = dst_offset});
        } else {
            read_source_page<Index + 1>(noc, dst_cb, input, page_id, dst_offset);
        }
    }
}

void kernel_main() {
    uint32_t num_pages = get_arg_val<uint32_t>(0);
    uint32_t src_page = get_arg_val<uint32_t>(1);

    Noc noc;
    CircularBuffer out_cb(cb_out);
    CircularBuffer scratch_cb(cb_scratch);

    // Private staging: only this RISC touches it, so there is nothing to reserve or push.
    const uint32_t scratch_addr = scratch_cb.get_write_ptr();

    while (num_pages > 0) {
        const uint32_t batch = num_pages < BATCH ? num_pages : BATCH;
        out_cb.reserve_back(batch);
        const uint32_t out_base = out_cb.get_write_ptr();

        // Address and width of the batch's final RISC store, 0 if it made none.
        uint32_t drain_addr = 0;
        bool drain_wide = false;

        for (uint32_t page = 0; page < batch; ++page) {
            uint32_t dst = out_base + page * OUT_PAGE_SIZE;
            uint32_t dst_offset = 0;

            for (uint32_t input = 0; input < N_INPUTS; ++input) {
                const uint32_t stick_size = get_common_arg_val<uint32_t>(stick_rt + input);
                const uint32_t page_size = get_common_arg_val<uint32_t>(page_rt + input);

                const uint32_t out_cb_offset = page * OUT_PAGE_SIZE + dst_offset;
                if (stick_size == page_size && (out_cb_offset % SRC_NOC_ALIGNMENT) == 0) {
                    read_source_page<0>(noc, out_cb, input, src_page + page, out_cb_offset);
                    dst_offset += stick_size;
                    continue;
                }

                read_source_page<0>(noc, scratch_cb, input, src_page + page, 0);
                noc.async_read_barrier();
                // A word store needs a word-aligned destination; sticks are a multiple of 2 bytes, not of 4.
                const bool word_store = (dst_offset & 3U) == 0;
                if (word_store) {
                    CoreLocalMem<volatile uint32_t> src(scratch_addr);
                    CoreLocalMem<volatile uint32_t> target(dst + dst_offset);
                    const uint32_t words = stick_size / 4;
                    for (uint32_t word = 0; word < words; ++word) {
                        target[word] = src[word];
                    }
                    if ((stick_size & 2U) != 0) {
                        CoreLocalMem<volatile uint16_t> src_tail(scratch_addr);
                        CoreLocalMem<volatile uint16_t> target_tail(dst + dst_offset);
                        target_tail[words * 2] = src_tail[words * 2];
                    }
                } else {
                    CoreLocalMem<volatile uint16_t> src(scratch_addr);
                    CoreLocalMem<volatile uint16_t> target(dst + dst_offset);
                    for (uint32_t word = 0; word < stick_size / 2; ++word) {
                        target[word] = src[word];
                    }
                }
                const bool ends_on_word = word_store && (stick_size & 2U) == 0;
                drain_wide = ends_on_word;
                drain_addr = dst + dst_offset + stick_size - (ends_on_word ? sizeof(uint32_t) : sizeof(uint16_t));
                dst_offset += stick_size;
            }
        }

        noc.async_read_barrier();
        // Staged-copy drain: RISC stores can retire after push_back, so the consumer could read the page before
        // they land. Reading the last stored address back blocks until they have reached L1.
        if (drain_addr != 0) {
            if (drain_wide) {
                (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(drain_addr));
            } else {
                (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(drain_addr));
            }
        }
        out_cb.push_back(batch);
        src_page += batch;
        num_pages -= batch;
    }
}
