// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "dprint.h"

#include "../scatter_common.hpp"

namespace {

template <bool is_dram, typename AddrGen>
FORCE_INLINE void load_to_cb(
    const uint32_t& cb, const AddrGen& addr_gtor, const uint32_t& stick_size_bytes, const uint32_t& stick_id) {
    cb_reserve_back(cb, ONE_PAGE);  // read a whole input row
    const uint64_t source_noc_address = get_noc_addr(stick_id, addr_gtor);
    const uint32_t l1_write_address = get_write_ptr(cb);

    noc_async_read(source_noc_address, l1_write_address, stick_size_bytes);
    noc_async_read_barrier();

    cb_push_back(cb, ONE_PAGE);
}

FORCE_INLINE void prepare_cbs(
    const uint32_t& input_cb, const uint32_t& index_cb, const uint32_t& source_cb, const uint32_t& output_cb) {
    cb_wait_front(input_cb, ONE_PAGE);
    cb_wait_front(index_cb, ONE_PAGE);
    cb_wait_front(source_cb, ONE_PAGE);
    cb_reserve_back(output_cb, ONE_PAGE);
}

FORCE_INLINE void recycle_input_and_push_result_row(
    const uint32_t input_cb, const uint32_t& index_cb, const uint32_t& source_cb, const uint32_t& output_cb) {
    cb_pop_front(input_cb, ONE_PAGE);
    cb_pop_front(index_cb, ONE_PAGE);
    cb_pop_front(source_cb, ONE_PAGE);
    cb_push_back(output_cb, ONE_PAGE);
}

template <typename number_type, typename index_type>
FORCE_INLINE void scatter_along_whole_axis(
    const uint32_t& input_cb,
    const uint32_t& index_cb,
    const uint32_t& source_cb,
    const uint32_t& output_cb,
    const uint32_t& input_stick_size,
    const uint32_t& index_stick_size,
    const uint32_t& source_stick_size,
    const uint32_t& output_stick_size) {
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

    for (uint32_t index_in_input_stick = 0; index_in_input_stick < input_stick_size; ++index_in_input_stick) {
        // DPRINT << "INPUT " << input_l1_read_ptr[index_in_input_stick] << ENDL();
        output_l1_write_ptr[index_in_input_stick] = input_l1_read_ptr[index_in_input_stick];
    }

    for (uint32_t index_in_index_stick = 0; index_in_index_stick < index_stick_size; ++index_in_index_stick) {
        volatile index_type& index_value = index_l1_read_ptr[index_in_index_stick];
        // DPRINT << "INDEX " << index_value << ENDL();
        ASSERT(
            index_value < index_stick_size,
            "Index value {} is bigger than dimension size {}.",
            index_value,
            index_stick_size);
        if (index_value >= index_stick_size) {
            continue;
        }
        volatile number_type& source_value = source_l1_read_ptr[index_in_index_stick];
        // DPRINT << "SOURCE " << source_value << ENDL();
        output_l1_write_ptr[index_value] = source_value;
    }

    for (uint32_t i = 0; i < input_stick_size; ++i) {
        DPRINT << i << " " << output_l1_write_ptr[i] << " " << ENDL();
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
        // load input sticks (rows)
        load_to_cb<ctas.input_tensor_is_dram, input_addr_gtor_type>(
            ctas.input_cb, input_addr_gtor, ctas.input_stick_size_bytes, stick_id);
        load_to_cb<ctas.index_tensor_is_dram, index_addr_gtor_type>(
            ctas.index_cb, index_addr_gtor, ctas.index_stick_size_bytes, stick_id);
        load_to_cb<ctas.source_tensor_is_dram, source_addr_gtor_type>(
            ctas.source_cb, source_addr_gtor, ctas.source_stick_size_bytes, stick_id);
        prepare_cbs(ctas.input_cb, ctas.index_cb, ctas.source_cb, ctas.output_cb);

        // scatter glitter yayy
        scatter_along_whole_axis<input_std_type, index_std_type>(
            ctas.input_cb,
            ctas.index_cb,
            ctas.source_cb,
            ctas.output_cb,
            ctas.input_stick_size,
            ctas.index_stick_size,
            ctas.source_stick_size,
            ctas.output_stick_size);

        // ecological and responsible
        recycle_input_and_push_result_row(ctas.input_cb, ctas.index_cb, ctas.source_cb, ctas.output_cb);
    }
}
