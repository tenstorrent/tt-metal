// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

template <
    typename elements_accessor_args_type,
    typename test_elements_accessor_args_type,
    typename output_accessor_args_type>
struct IsInCTAs {
    const uint32_t elements_tensor_addr;
    const uint32_t test_elements_tensor_addr;
    const uint32_t output_tensor_addr;
    const uint32_t elements_cb;
    const uint32_t test_elements_cb;
    const uint32_t output_cb;
    const uint32_t elements_size;
    const uint32_t test_elements_size;
    const uint32_t single_fetch_subchunk_size;
    const bool invert;
    const uint32_t elements_tensor_datum_size;
    const elements_accessor_args_type elements_accessor_args;
    const test_elements_accessor_args_type test_elements_accessor_args;
    const output_accessor_args_type output_accessor_args;
};

// get compile-time arguments by any kernel
FORCE_INLINE constexpr auto get_ctas() {
    constexpr auto elements_args = TensorAccessorArgs<11>();
    constexpr auto test_elements_args = TensorAccessorArgs<elements_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<test_elements_args.next_compile_time_args_offset()>();
    return IsInCTAs<decltype(elements_args), decltype(test_elements_args), decltype(output_args)>{
        get_compile_time_arg_val(0),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9) != 0,
        get_compile_time_arg_val(10),
        elements_args,
        test_elements_args,
        output_args};
}

// load from DRAM to L1
template <typename addr_gen_type>
FORCE_INLINE void load_to_cb(
    const uint32_t& cb,
    const addr_gen_type& addr_gtor,
    const uint32_t& offset,
    const uint32_t& subchunk_size,
    const uint32_t& datum_size) {
    cb_reserve_back(cb, ONE_PAGE);

    const uint64_t source_noc_address = get_noc_addr(FIRST_STICK, addr_gtor);
    const uint32_t l1_write_address = get_write_ptr(cb);
    const uint32_t subchunk_size_bytes = subchunk_size * datum_size;
    const uint32_t offset_bytes = offset * datum_size;
    noc_async_read(source_noc_address + offset_bytes, l1_write_address, subchunk_size_bytes);
    noc_async_read_barrier();

    cb_push_back(cb, ONE_PAGE);
}

// write from L1 to DRAM
template <typename addr_gen_type>
FORCE_INLINE void write_to_dram(
    const uint32_t& cb,
    const addr_gen_type& addr_gtor,
    const uint32_t& offset,
    const uint32_t& subchunk_size,
    const uint32_t& datum_size) {
    cb_wait_front(cb, ONE_PAGE);

    const uint64_t destination_noc_address = get_noc_addr(FIRST_STICK, addr_gtor);
    const uint32_t l1_read_address = get_read_ptr(cb);
    const uint32_t subchunk_size_bytes = subchunk_size * datum_size;
    const uint32_t offset_bytes = offset * datum_size;
    noc_async_write(l1_read_address, destination_noc_address + offset_bytes, subchunk_size_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb, ONE_PAGE);
}
