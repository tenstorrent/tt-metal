// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

using elements_number_type = uint32_t;
using output_number_type = uint32_t;

// choose the right C++ POD type at compile-time
template <DataFormat df>
struct df_to_std {
    using std_type = void;
};

template <>
struct df_to_std<DataFormat::Float32> {
    using std_type = float;
};

template <>
struct df_to_std<DataFormat::Float16_b> {
    using std_type = uint16_t;
};

template <>
struct df_to_std<DataFormat::Int32> {
    using std_type = uint32_t;
};

template <>
struct df_to_std<DataFormat::UInt32> {
    using std_type = uint32_t;
};

template <>
struct df_to_std<DataFormat::UInt16> {
    using std_type = uint16_t;
};

template <>
struct df_to_std<DataFormat::UInt8> {
    using std_type = uint8_t;
};

template <DataFormat df>
using std_type_t = typename df_to_std<df>::std_type;

constexpr uint32_t ONE_PAGE = 1;
constexpr uint32_t FIRST_STICK = 0;
constexpr uint32_t OUTPUT_SIZE_TENSOR_SIZE = 64;

template <
    typename input_accessor_args_type,
    typename first_occurrences_accessor_args_type,
    typename output_accessor_args_type,
    typename output_size_accessor_args_type>
struct UniqueCTAs {
    const uint32_t input_tensor_addr;
    const uint32_t first_occurrences_tensor_addr;
    const uint32_t output_tensor_addr;
    const uint32_t output_size_tensor_addr;
    const uint32_t input_cb;
    const uint32_t input_compare_cb;
    const uint32_t first_occurrences_read_cb;
    const uint32_t first_occurrences_write_cb;
    const uint32_t first_occurrences_output_cb;
    const uint32_t result_cb;
    const uint32_t output_cb;
    const uint32_t output_size_cb;
    const uint32_t input_size;
    const uint32_t single_fetch_subchunk_size;
    const input_accessor_args_type input_accessor_args;
    const first_occurrences_accessor_args_type first_occurrences_accessor_args;
    const output_accessor_args_type output_accessor_args;
    const output_size_accessor_args_type output_size_accessor_args;
};

FORCE_INLINE constexpr auto get_ctas() {
    constexpr auto input_args = TensorAccessorArgs<14>();
    constexpr auto first_occurrences_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<first_occurrences_args.next_compile_time_args_offset()>();
    constexpr auto output_size_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    return UniqueCTAs<
        decltype(input_args),
        decltype(first_occurrences_args),
        decltype(output_args),
        decltype(output_size_args)>{
        get_compile_time_arg_val(0),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        input_args,
        first_occurrences_args,
        output_args,
        output_size_args};
}

template <typename addr_gen_type>
FORCE_INLINE void load_to_cb(
    uint32_t cb,
    const addr_gen_type& addr_gtor,
    uint32_t offset,
    uint32_t subchunk_size,
    uint32_t input_element_size = 4) {
    cb_reserve_back(cb, ONE_PAGE);

    const uint64_t source_noc_address = get_noc_addr(FIRST_STICK, addr_gtor);
    const uint32_t l1_write_address = get_write_ptr(cb);
    const uint32_t subchunk_size_bytes = subchunk_size * input_element_size;
    const uint32_t offset_bytes = offset * input_element_size;
    noc_async_read(source_noc_address + offset_bytes, l1_write_address, subchunk_size_bytes);
    noc_async_read_barrier();

    cb_push_back(cb, ONE_PAGE);
}

template <typename addr_gen_type>
FORCE_INLINE void write_to_dram(
    uint32_t cb,
    const addr_gen_type& addr_gtor,
    uint32_t offset,
    uint32_t subchunk_size,
    uint32_t output_element_size = 4) {
    cb_wait_front(cb, ONE_PAGE);

    const uint64_t destination_noc_address = get_noc_addr(FIRST_STICK, addr_gtor);
    const uint32_t l1_read_address = get_read_ptr(cb);
    const uint32_t subchunk_size_bytes = subchunk_size * output_element_size;
    const uint32_t offset_bytes = offset * output_element_size;
    noc_async_write(l1_read_address, destination_noc_address + offset_bytes, subchunk_size_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb, ONE_PAGE);
}
