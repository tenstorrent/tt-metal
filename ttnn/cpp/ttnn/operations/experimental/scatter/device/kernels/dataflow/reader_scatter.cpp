// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "dprint.h"

#include "../scatter_common.hpp"

namespace {

// this function is supposed to load either a whole stick or part of it (76800 elements)
template <bool is_dram, typename AddrGen>
FORCE_INLINE void load_to_cb(
    const uint32_t& cb, const AddrGen& addr_gtor, const uint32_t& offset_bytes, const uint32_t& chunk_size_bytes, const uint32_t& stick_id) {
    cb_reserve_back(cb, ONE_PAGE);
    const uint64_t source_noc_address = get_noc_addr(stick_id, addr_gtor);
    const uint32_t l1_write_address = get_write_ptr(cb);

    noc_async_read(source_noc_address + offset_bytes, l1_write_address, chunk_size_bytes);
    noc_async_read_barrier();

    cb_push_back(cb, ONE_PAGE);
}

// copies source stick to destination stick (first phase of scatter)
template <typename number_type>
FORCE_INLINE void copy_input_to_output(
    const uint32_t& input_cb, const uint32_t& output_cb, const uint32_t& input_chunk_size) {
    const uint32_t input_l1_read_addr = get_read_ptr(input_cb);
    const uint32_t output_l1_write_addr = get_write_ptr(output_cb);
    volatile tt_l1_ptr number_type* input_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(input_l1_read_addr);
    volatile tt_l1_ptr number_type* output_l1_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(output_l1_write_addr);
    for (uint32_t index_in_input_chunk = 0; index_in_input_chunk < input_chunk_size; ++index_in_input_chunk) {
        output_l1_write_ptr[index_in_input_chunk] = input_l1_read_ptr[index_in_input_chunk];
    }
}

// performs scatter on data loaded to cb with load_to_cb
template <typename number_type, typename index_type>
FORCE_INLINE void scatter_along_chunk(
    const uint32_t& input_cb,
    const uint32_t& index_cb,
    const uint32_t& source_cb,
    const uint32_t& output_cb,
    const uint32_t& input_stick_size,
    const uint32_t& input_offset,
    const uint32_t& input_chunk_size,
    const uint32_t& index_chunk_size) {
    const uint32_t input_l1_read_addr = get_read_ptr(input_cb);
    const uint32_t index_l1_read_addr = get_read_ptr(index_cb);
    const uint32_t source_l1_read_addr = get_read_ptr(source_cb);
    const uint32_t output_l1_write_addr = get_write_ptr(output_cb);
    volatile tt_l1_ptr number_type* input_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(input_l1_read_addr);
    volatile tt_l1_ptr index_type* index_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr index_type*>(index_l1_read_addr);
    volatile tt_l1_ptr number_type* source_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(source_l1_read_addr);
    volatile tt_l1_ptr number_type* output_l1_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(output_l1_write_addr);

    // each index from the index chunk is checked whether it points
    // to any of the elements in the current output range (defined by
    // partial stick length and offset)
    for (uint32_t index_in_index_chunk = 0; index_in_index_chunk < index_chunk_size; ++index_in_index_chunk) {
        volatile index_type& index_value = index_l1_read_ptr[index_in_index_chunk];
        if (index_value < input_offset || index_value >= input_offset + input_chunk_size) {
            continue;
        }
        ASSERT(
            index_value < input_stick_size,
            "Index value {} is bigger than input's dimension size {}.",
            index_value,
            input_stick_size);
        if (index_value >= input_stick_size) {
            continue;
        }
        volatile number_type& source_value = source_l1_read_ptr[index_in_index_chunk];
        output_l1_write_ptr[index_value - input_offset] = source_value;
    }
}

}  // namespace

void kernel_main() {
    constexpr auto ctas{get_ctas()};

    const uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t index_buffer_address = get_arg_val<uint32_t>(1);
    const uint32_t source_buffer_address = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    const uint32_t sticks_for_core = get_arg_val<uint32_t>(4);
    // for the outer input/output loop (DRAM accesses per stick: input_row_elem_num / 76800)
    const uint32_t input_and_output_chunk_size = get_arg_val<uint32_t>(5);
    // for the inner index/source loop (DRAM accesses per stick per single input/output loop: index_row_elem_num /
    // 76800)
    const uint32_t index_and_source_chunk_size = get_arg_val<uint32_t>(6);

    const auto input_addr_gtor{
        get_interleaved_addr_gen<ctas.input_tensor_is_dram, ctas.is_input_stick_size_bytes_pow2_min_32>(
            input_buffer_address, ctas.input_stick_size_bytes, ctas.input_stick_size_bytes_log2)};
    const auto index_addr_gtor{
        get_interleaved_addr_gen<ctas.index_tensor_is_dram, ctas.is_index_stick_size_bytes_pow2_min_32>(
            index_buffer_address, ctas.index_stick_size_bytes, ctas.index_stick_size_bytes_log2)};
    const auto source_addr_gtor{
        get_interleaved_addr_gen<ctas.source_tensor_is_dram, ctas.is_source_stick_size_bytes_pow2_min_32>(
            source_buffer_address, ctas.source_stick_size_bytes, ctas.source_stick_size_bytes_log2)};

    using input_std_type = std_type_t<get_dataformat(ctas.input_cb)>;
    using index_std_type = std_type_t<get_dataformat(ctas.index_cb)>;
    using input_addr_gtor_type = decltype(input_addr_gtor);
    using index_addr_gtor_type = decltype(index_addr_gtor);
    using source_addr_gtor_type = decltype(source_addr_gtor);

    for (uint32_t stick_id = start_stick_id; stick_id < start_stick_id + sticks_for_core; ++stick_id) {
        // process input/output chunks sequentially
        for (uint32_t input_offset = 0; input_offset < ctas.input_stick_size;
             input_offset += input_and_output_chunk_size) {
            const uint32_t input_chunk_length =
                std::min(ctas.input_stick_size - input_offset, input_and_output_chunk_size);

            // first phase: copy input data to output
            load_to_cb<ctas.input_tensor_is_dram, input_addr_gtor_type>(
                ctas.input_cb,
                input_addr_gtor,
                input_offset * sizeof(input_std_type),
                input_chunk_length * sizeof(input_std_type),
                stick_id);
            cb_wait_front(ctas.input_cb, ONE_PAGE);
            cb_reserve_back(ctas.output_cb, ONE_PAGE);

            copy_input_to_output<input_std_type>(ctas.input_cb, ctas.output_cb, input_chunk_length);

            // second phase: load index and source data chunk-by-chunk and scatter
            for (uint32_t index_offset = 0; index_offset < ctas.index_stick_size;
                 index_offset += index_and_source_chunk_size) {
                // if stick is chunked, the last chunk is usually smaller
                const uint32_t index_chunk_length =
                    std::min(ctas.index_stick_size - index_offset, index_and_source_chunk_size);
                load_to_cb<ctas.index_tensor_is_dram, index_addr_gtor_type>(
                    ctas.index_cb,
                    index_addr_gtor,
                    index_offset * sizeof(index_std_type),
                    index_chunk_length * sizeof(index_std_type),
                    stick_id);
                load_to_cb<ctas.source_tensor_is_dram, source_addr_gtor_type>(
                    ctas.source_cb,
                    source_addr_gtor,
                    index_offset * sizeof(input_std_type),
                    index_chunk_length * sizeof(input_std_type),
                    stick_id);
                cb_wait_front(ctas.index_cb, ONE_PAGE);
                cb_wait_front(ctas.source_cb, ONE_PAGE);
                scatter_along_chunk<input_std_type, index_std_type>(
                    ctas.input_cb,
                    ctas.index_cb,
                    ctas.source_cb,
                    ctas.output_cb,
                    ctas.input_stick_size,
                    input_offset,
                    input_chunk_length,
                    index_chunk_length);
                cb_pop_front(ctas.source_cb, ONE_PAGE);
                cb_pop_front(ctas.index_cb, ONE_PAGE);
            }

            // third phase: push to the output cb
            cb_push_back(ctas.output_cb, ONE_PAGE);
            cb_pop_front(ctas.input_cb, ONE_PAGE);
        }
    }
}
