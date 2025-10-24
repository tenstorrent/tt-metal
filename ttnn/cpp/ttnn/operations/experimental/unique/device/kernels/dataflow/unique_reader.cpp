
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include "unique_common.hpp"

#include <algorithm>
#include <numeric>

namespace {

template <typename input_number_type>
FORCE_INLINE void copy_input_cb_to_output_cb(
    uint32_t input_l1_read_addr, uint32_t output_l1_write_addr, uint32_t subchunk_size) {
    volatile tt_l1_ptr input_number_type* input_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_number_type*>(input_l1_read_addr);
    volatile tt_l1_ptr input_number_type* output_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_number_type*>(output_l1_write_addr);
    for (uint32_t i = 0; i < subchunk_size; ++i) {
        output_subchunk_ptr[i] = input_subchunk_ptr[i];
    }
}

template <typename input_number_type, typename first_occurrences_number_type>
FORCE_INLINE void unique_subchunks(
    uint32_t input_l1_read_addr,
    uint32_t input_compare_l1_read_addr,
    uint32_t first_occurrences_l1_read_addr,
    uint32_t first_occurrences_l1_write_addr,
    // uint32_t output_l1_write_addr,
    uint32_t starting_input_global_index,
    uint32_t starting_input_compare_global_index,
    uint32_t input_subchunk_size,
    uint32_t input_compare_subchunk_size) {
    volatile tt_l1_ptr input_number_type* input_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_number_type*>(input_l1_read_addr);
    volatile tt_l1_ptr input_number_type* input_compare_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_number_type*>(input_compare_l1_read_addr);
    volatile tt_l1_ptr first_occurrences_number_type* first_occurrences_read_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr first_occurrences_number_type*>(first_occurrences_l1_read_addr);
    volatile tt_l1_ptr first_occurrences_number_type* first_occurrences_write_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr first_occurrences_number_type*>(first_occurrences_l1_write_addr);
    // for (uint32_t input_index_inverted = input_subchunk_size - 1,
    //               input_global_index_inverted = starting_input_global_index + input_subchunk_size - 1;
    //      input_index >= 0;
    //      --input_index, --input_global_index) {
    for (uint32_t input_index_inverted = 0; input_index_inverted < input_subchunk_size - 1; ++input_index_inverted) {
        const uint32_t input_index = input_subchunk_size - 1 - input_index_inverted;
        const uint32_t input_global_index =
            starting_input_global_index + input_subchunk_size - 1 - input_index_inverted;
        const uint32_t ending_input_compare_index = (starting_input_global_index == starting_input_compare_global_index)
                                                        ? input_index
                                                        : input_compare_subchunk_size;
        if (first_occurrences_read_subchunk_ptr[input_index] == input_global_index) {
            bool updated = false;
            for (uint32_t input_compare_index = 0, input_compare_global_index = starting_input_compare_global_index;
                 input_compare_index < ending_input_compare_index;
                 ++input_compare_index, ++input_compare_global_index) {
                if (input_subchunk_ptr[input_index] == input_compare_subchunk_ptr[input_compare_index]) {
                    first_occurrences_write_subchunk_ptr[input_index] = input_compare_global_index;
                    updated = true;
                    break;
                }
            }
            if (!updated) {
                first_occurrences_write_subchunk_ptr[input_index] = first_occurrences_read_subchunk_ptr[input_index];
            }
        }
        DPRINT << "input_index " << input_index << ENDL();
    }
}

template <typename first_occurrences_number_type, typename first_occurrences_addr_gtor_type>
FORCE_INLINE void prefill_first_occurrences(
    uint32_t first_occurrences_write_cb,
    const first_occurrences_addr_gtor_type& first_occurrences_addr_gtor,
    uint32_t starting_global_index,
    uint32_t first_occurrences_subchunk_size,
    uint32_t first_occurrences_element_size = 4) {
    cb_reserve_back(first_occurrences_write_cb, ONE_PAGE);
    const uint32_t first_occurrences_write_l1_subchunk_addr = get_write_ptr(first_occurrences_write_cb);
    volatile tt_l1_ptr first_occurrences_number_type* first_occurrences_write_subchunk_begin_ptr =
        reinterpret_cast<volatile tt_l1_ptr first_occurrences_number_type*>(first_occurrences_write_l1_subchunk_addr);
    for (uint32_t index = 0, global_index = starting_global_index; index < first_occurrences_subchunk_size;
         ++index, ++global_index) {
        first_occurrences_write_subchunk_begin_ptr[index] = global_index;
    }
    cb_push_back(first_occurrences_write_cb, ONE_PAGE);
    write_to_dram(
        first_occurrences_write_cb,
        first_occurrences_addr_gtor,
        starting_global_index,
        first_occurrences_subchunk_size,
        first_occurrences_element_size);
}

}  // namespace

void kernel_main() {
    constexpr auto ctas = get_ctas();

    const uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t first_occurrences_buffer_address = get_arg_val<uint32_t>(1);
    const uint32_t subchunks_per_core = get_arg_val<uint32_t>(2);
    const uint32_t subchunks_offset = get_arg_val<uint32_t>(3);

    using input_number_type = std_type_t<get_dataformat(ctas.input_cb)>;
    using first_occurrences_number_type = std_type_t<get_dataformat(ctas.first_occurrences_read_cb)>;

    const auto input_addr_gtor =
        TensorAccessor(ctas.input_accessor_args, input_buffer_address, ctas.input_size * sizeof(input_number_type));
    const auto first_occurrences_addr_gtor = TensorAccessor(
        ctas.first_occurrences_accessor_args,
        first_occurrences_buffer_address,
        ctas.input_size * sizeof(first_occurrences_number_type));

    for (uint32_t input_subchunk_id = subchunks_offset,
                  input_offset = subchunks_offset * ctas.single_fetch_subchunk_size;
         input_subchunk_id < subchunks_offset + subchunks_per_core;
         ++input_subchunk_id, input_offset += ctas.single_fetch_subchunk_size) {
        const uint32_t input_subchunk_size = std::min(ctas.input_size - input_offset, ctas.single_fetch_subchunk_size);
        prefill_first_occurrences<first_occurrences_number_type, decltype(first_occurrences_addr_gtor)>(
            ctas.first_occurrences_write_cb, first_occurrences_addr_gtor, input_offset, input_subchunk_size);
        load_to_cb(ctas.input_cb, input_addr_gtor, input_offset, input_subchunk_size);
        cb_wait_front(ctas.input_cb, ONE_PAGE);
        const uint32_t input_l1_read_addr = get_read_ptr(ctas.input_cb);
        cb_reserve_back(ctas.output_cb, ONE_PAGE);

        for (uint32_t input_compare_subchunk_id = 0, input_compare_offset = 0;
             input_compare_subchunk_id <= input_subchunk_id;
             ++input_compare_subchunk_id, input_compare_offset += ctas.single_fetch_subchunk_size) {
            load_to_cb(ctas.first_occurrences_read_cb, first_occurrences_addr_gtor, input_offset, input_subchunk_size);
            cb_reserve_back(ctas.first_occurrences_write_cb, ONE_PAGE);
            const uint32_t first_occurrences_l1_write_addr = get_write_ptr(ctas.first_occurrences_write_cb);
            cb_wait_front(ctas.first_occurrences_read_cb, ONE_PAGE);
            const uint32_t first_occurrences_l1_read_addr = get_read_ptr(ctas.first_occurrences_read_cb);

            const uint32_t input_compare_subchunk_size =
                std::min(ctas.input_size - input_compare_offset, ctas.single_fetch_subchunk_size);
            load_to_cb(ctas.input_compare_cb, input_addr_gtor, input_compare_offset, input_compare_subchunk_size);
            cb_wait_front(ctas.input_compare_cb, ONE_PAGE);
            const uint32_t input_compare_l1_read_addr = get_read_ptr(ctas.input_compare_cb);

            unique_subchunks<input_number_type, first_occurrences_number_type>(
                input_l1_read_addr,
                input_compare_l1_read_addr,
                first_occurrences_l1_read_addr,
                first_occurrences_l1_write_addr,
                input_offset,
                input_compare_offset,
                input_subchunk_size,
                input_compare_subchunk_size);

            cb_push_back(ctas.first_occurrences_write_cb, ONE_PAGE);
            write_to_dram(
                ctas.first_occurrences_write_cb,
                first_occurrences_addr_gtor,
                input_compare_offset,
                input_compare_subchunk_size);
            cb_pop_front(ctas.input_compare_cb, ONE_PAGE);
            cb_pop_front(ctas.first_occurrences_read_cb, ONE_PAGE);
        }

        cb_reserve_back(ctas.output_cb, ONE_PAGE);
        const uint32_t output_l1_write_addr = get_write_ptr(ctas.output_cb);
        copy_input_cb_to_output_cb<input_number_type>(input_l1_read_addr, output_l1_write_addr, input_subchunk_size);
        cb_push_back(ctas.output_cb, ONE_PAGE);
        load_to_cb(ctas.first_occurrences_output_cb, first_occurrences_addr_gtor, input_offset, input_subchunk_size);
        cb_pop_front(ctas.input_cb, ONE_PAGE);
    }
}
