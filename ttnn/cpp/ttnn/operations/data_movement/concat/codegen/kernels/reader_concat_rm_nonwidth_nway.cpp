// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// N-way non-width reader: every output stick comes whole from one input, so a cursor walks per-input blocks.
#include <tuple>
#include <type_traits>
#include <utility>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_id = get_compile_time_arg_val(0);
constexpr uint32_t BATCH = get_compile_time_arg_val(1);
constexpr uint32_t NUM_INPUTS = get_compile_time_arg_val(2);
constexpr uint32_t CB_PAGE_SIZE = get_compile_time_arg_val(3);
constexpr uint32_t FIRST_PAGE_BYTES_ARG = 4;
constexpr uint32_t FIRST_ACCESSOR_ARGS = FIRST_PAGE_BYTES_ARG + NUM_INPUTS;

constexpr uint32_t BASE_OFFSET = 0;
constexpr uint32_t BLOCK_OFFSET = BASE_OFFSET + NUM_INPUTS;
constexpr uint32_t STICK_ID_OFFSET = 3;

constexpr auto source_args = make_tensor_accessor_args_tuple<NUM_INPUTS, FIRST_ACCESSOR_ARGS>();

using SourceArgsTuple = std::remove_const_t<decltype(source_args)>;

template <size_t... Index>
constexpr bool same_bank_type(std::index_sequence<Index...>) {
    return (
        (std::tuple_element_t<Index, SourceArgsTuple>::is_dram == std::tuple_element_t<0, SourceArgsTuple>::is_dram) &&
        ...);
}

// Every input has the same width, so with a common bank type they share one accessor and page size.
constexpr bool UNIFORM_PLACEMENT = same_bank_type(std::make_index_sequence<NUM_INPUTS>{});

// Mixed DRAM/L1 inputs carry distinct accessor types, so the one to read through is picked by walking the inputs.
template <uint32_t Index>
FORCE_INLINE void read_stick(Noc& noc, CircularBuffer& cb, uint32_t input, uint32_t stick_id, uint32_t l1_offset) {
    if constexpr (UNIFORM_PLACEMENT) {
        constexpr auto args = std::get<0>(source_args);
        constexpr uint32_t page_bytes = get_compile_time_arg_val(FIRST_PAGE_BYTES_ARG);
        const auto source = TensorAccessor(args, get_common_arg_val<uint32_t>(BASE_OFFSET + input), page_bytes);
        noc.async_read(source, cb, page_bytes, {.page_id = stick_id}, {.offset_bytes = l1_offset});
    } else if constexpr (Index < NUM_INPUTS) {
        if (input == Index) {
            constexpr auto args = std::get<Index>(source_args);
            // Compile-time so the accessor's page arithmetic and the read size fold to constants.
            constexpr uint32_t page_bytes = get_compile_time_arg_val(FIRST_PAGE_BYTES_ARG + Index);
            const auto source = TensorAccessor(args, get_common_arg_val<uint32_t>(BASE_OFFSET + Index), page_bytes);
            noc.async_read(source, cb, page_bytes, {.page_id = stick_id}, {.offset_bytes = l1_offset});
        } else {
            read_stick<Index + 1>(noc, cb, input, stick_id, l1_offset);
        }
    }
}

void kernel_main() {
    uint32_t num_sticks = get_arg_val<uint32_t>(0);
    uint32_t current_input = get_arg_val<uint32_t>(1);
    uint32_t current_input_stick = get_arg_val<uint32_t>(2);

    uint32_t sticks_per_block[NUM_INPUTS];
    uint32_t stick_ids[NUM_INPUTS];
    for (uint32_t input = 0; input < NUM_INPUTS; ++input) {
        sticks_per_block[input] = get_common_arg_val<uint32_t>(BLOCK_OFFSET + input);
        stick_ids[input] = get_arg_val<uint32_t>(STICK_ID_OFFSET + input);
    }

    Noc noc;
    CircularBuffer input_cb(cb_id);

    uint32_t sticks_left = num_sticks;
    while (sticks_left > 0) {
        const uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        input_cb.reserve_back(batch);
        uint32_t l1_offset = 0;

        for (uint32_t stick = 0; stick < batch; ++stick) {
            read_stick<0>(noc, input_cb, current_input, stick_ids[current_input]++, l1_offset);
            l1_offset += CB_PAGE_SIZE;

            current_input_stick++;
            if (current_input_stick == sticks_per_block[current_input]) {
                current_input_stick = 0;
                current_input++;
                if (current_input == NUM_INPUTS) {
                    current_input = 0;
                }
            }
        }
        noc.async_read_barrier();
        input_cb.push_back(batch);
        sticks_left -= batch;
    }
}
