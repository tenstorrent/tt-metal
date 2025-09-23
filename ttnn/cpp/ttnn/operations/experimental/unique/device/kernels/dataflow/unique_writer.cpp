// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "unique_common.hpp"

void kernel_main() {
    constexpr auto ctas = get_ctas();

    const uint32_t first_occurrences_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t output_buffer_address = get_arg_val<uint32_t>(1);
    const uint32_t output_size_buffer_address = get_arg_val<uint32_t>(2);
    const uint32_t subchunks_per_core = get_arg_val<uint32_t>(3);
    const uint32_t subchunks_offset = get_arg_val<uint32_t>(4);
    const auto output_addr_gtor =
        TensorAccessor{ctas.output_accessor_args, output_buffer_address, ctas.input_size * ctas.input_element_size};
    const auto output_size_addr_gtor = TensorAccessor{ctas.output_size_accessor_args, output_size_buffer_address, 64UL};
    const auto first_occurrences_addr_gtor = TensorAccessor{
        ctas.first_occurrences_accessor_args,
        first_occurrences_buffer_address,
        ctas.input_size * ctas.first_occurrences_element_size};

    using output_number_type = std_type_t<get_dataformat(ctas.output_cb)>;
    using first_occurrences_number_type = std_type_t<get_dataformat(ctas.first_occurrences_read_cb)>;

    uint32_t result_index = 0;
    uint32_t global_output_index = 0;
    cb_reserve_back(ctas.result_cb, ONE_PAGE);
    uint32_t result_l1_write_address = get_write_ptr(ctas.result_cb);
    for (uint32_t output_subchunk_id = subchunks_offset,
                  output_offset = subchunks_offset * ctas.single_fetch_subchunk_size;
         output_subchunk_id < subchunks_offset + subchunks_per_core;
         ++output_subchunk_id, output_offset += ctas.single_fetch_subchunk_size) {
        const uint32_t output_subchunk_size =
            std::min(ctas.input_size - output_offset, ctas.single_fetch_subchunk_size);
        load_to_cb(ctas.output_cb, ONE_PAGE);
        load_to_cb(ctas.first_occurrences_read_cb, ONE_PAGE);
        // const uint32_t output_l1_write_addr = get_write_ptr(ctas.output_cb);
        cb_wait_front(ctas.output_cb, ONE_PAGE);
        cb_wait_front(ctas.first_occurrences_read_cb, ONE_PAGE);
        const uint32_t output_l1_read_address = get_read_ptr(ctas.output_cb);
        const uint32_t first_occurrences_l1_read_address = get_read_ptr(ctas.first_occurrences_read_cb);
        volatile tt_l1_ptr output_number_type* output_subchunk_ptr =
            reinterpret_cast<volatile tt_l1_ptr output_number_type*>(output_l1_read_address);
        volatile tt_l1_ptr first_occurrences_number_type* first_occurrences_read_subchunk_ptr =
            reinterpret_cast<volatile tt_l1_ptr first_occurrences_number_type*>(first_occurrences_l1_read_address);
        volatile tt_l1_ptr output_number_type* result_write_subchunk_ptr =
            reinterpret_cast<volatile tt_l1_ptr output_number_type*>(result_l1_write_address);
        for (uint32_t output_index = 0; output_index < output_subchunk_size; ++output_index, ++global_output_index) {
            if (result_index < output_subchunk_size) {
                // act only if below occurs
                if (first_occurrences_read_subchunk_ptr[output_index] == global_output_index) {
                    // write to the result chunk and move the pointer
                    result_write_subchunk_ptr[result_index] = output_subchunk_ptr[output_index];
                }
            } else {
                // result -> output and reserve new result
                result_index = 0;
                cb_push_back(ctas.result_cb, ONE_PAGE);
                write_to_dram(
                    ctas.result_cb, output_addr_gtor, output_offset, output_subchunk_size, ctas.input_element_size);
                cb_reserve_back(ctas.result_cb, ONE_PAGE);
                uint32_t result_l1_write_address = get_write_ptr(ctas.result_cb);
                result_write_subchunk_ptr =
                    reinterpret_cast<volatile tt_l1_ptr output_number_type*>(result_l1_write_address);
            }
        }
    }
}
