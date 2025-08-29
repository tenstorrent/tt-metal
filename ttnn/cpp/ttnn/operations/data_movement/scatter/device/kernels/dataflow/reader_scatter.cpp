// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "dprint.h"

#include "../scatter_common.hpp"

namespace {

// this function is supposed to load either a whole stick or part of it (76800 elements)
template <typename AddrGen>
FORCE_INLINE void load_to_cb(
    const uint32_t& cb,
    const AddrGen& addr_gtor,
    const uint32_t& offset_bytes,
    const uint32_t& chunk_size_bytes,
    const uint32_t& stick_id) {
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

template <typename array_t, uint32_t rank>
FORCE_INLINE uint32_t get_index_stick_id(const array_t& index_slice, const array_t& index_shape) {
    uint32_t index_stick_id = 0;
    for (uint32_t dim = rank - 1; dim >= 0; --dim) {
        index_stick_id = index_stick_id * 10 + index_slice[dim];
    }
    return index_stick_id;
}

// template <typename array_t, uint32_t rank>
// FORCE_INLINE bool index_stick_in_input(const array_t& input_slice, const array_t& index_slice) {

// }

template <typename array_t, uint32_t rank>
FORCE_INLINE bool increment_index_stick(array_t& index_slice, const array_t& index_shape) {
    if (index_shape[rank - 1] - index_slice[rank - 1] > 1) {
        ++index_slice[rank - 1];
        return true;
    }
    bool changed = false;
    bool carry = false;
    for (uint32_t dim = rank - 1; dim >= 0; --dim) {
        if (index_shape[dim] - index_slice[dim] > 1) {
            ++index_slice[dim];
            break;
        } else {
            index_slicep[]
        }
    }

    return changed;
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
    const uint32_t index_chunk_size = get_arg_val<uint32_t>(6);
    const uint32_t source_chunk_size = get_arg_val<uint32_t>(7);

    // generate 2 shape arrays
    constexpr std::array<uint32_t, ctas.input_rank> input_shape{make_array<ctas.input_rank>(8)};
    constexpr std::array<uint32_t, ctas.input_rank> index_shape{make_array<ctas.input_rank>(8 + ctas.input_rank)};
    std::array<uint32_t, ctas.input_rank> current_input_slice_counter{};
    std::fill(current_input_slice_counter.begin(), current_input_slice_counter.emd(), 0);
    std::array<uint32_t, ctas.input_rank> current_index_slice_counter{};
    std::fill(current_index_slice_counter.begin(), current_index_slice_counter.emd(), 0);

    const auto input_addr_gtor = TensorAccessor(ctas.input_args, input_buffer_address, ctas.input_stick_size_bytes);
    const auto index_addr_gtor = TensorAccessor(ctas.index_args, index_buffer_address, ctas.index_stick_size_bytes);
    const auto source_addr_gtor = TensorAccessor(ctas.source_args, source_buffer_address, ctas.source_stick_size_bytes);

    using input_std_type = std_type_t<get_dataformat(ctas.input_cb)>;
    using index_std_type = std_type_t<get_dataformat(ctas.index_cb)>;

    for (uint32_t input_stick_id = start_stick_id; input_stick_id < start_stick_id + sticks_for_core;
         ++input_stick_id) {
        // process input/output chunks sequentially
        for (uint32_t input_offset = 0; input_offset < ctas.input_stick_size;
             input_offset += input_and_output_chunk_size) {
            const uint32_t input_chunk_length =
                std::min(ctas.input_stick_size - input_offset, input_and_output_chunk_size);

            // first phase: copy input data to output
            load_to_cb(
                ctas.input_cb,
                input_addr_gtor,
                input_offset * sizeof(input_std_type),
                input_chunk_length * sizeof(input_std_type),
                input_stick_id);
            cb_wait_front(ctas.input_cb, ONE_PAGE);
            cb_reserve_back(ctas.output_cb, ONE_PAGE);

            copy_input_to_output<input_std_type>(ctas.input_cb, ctas.output_cb, input_chunk_length);

            const index_stick_id = get_index_stick_id(current_index_slice_counter, index_shape);
            // second phase: load index and source data chunk-by-chunk and scatter
            for (uint32_t index_offset = 0, source_offset = 0; index_offset < ctas.index_stick_size;
                 index_offset += index_chunk_size, source_offset += source_chunk_size) {
                // if stick is chunked, the last chunk is usually smaller
                const uint32_t index_chunk_length = std::min(ctas.index_stick_size - index_offset, index_chunk_size);
                const uint32_t source_chunk_length =
                    std::min(ctas.source_stick_size - source_offset, source_chunk_size);
                load_to_cb(
                    ctas.index_cb,
                    index_addr_gtor,
                    index_offset * sizeof(index_std_type),
                    index_chunk_length * sizeof(index_std_type),
                    index_stick_id);
                load_to_cb(
                    ctas.source_cb,
                    source_addr_gtor,
                    source_offset * sizeof(input_std_type),
                    source_chunk_length * sizeof(input_std_type),
                    index_stick_id);
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
