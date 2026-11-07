// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

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

// Floating-point values are compared on their bits, the way torch compares floats: +0.0 equals -0.0
// and NaN equals nothing. bfloat16 is held as raw uint16 bits, and kernels build with -ffast-math, so
// neither the uint16 compare nor a float compare gives those semantics. Integer formats compare directly.
template <typename bits_type, bits_type magnitude_mask, bits_type positive_inf>
FORCE_INLINE bool float_bits_equal(bits_type a, bits_type b) {
    const bits_type a_magnitude = a & magnitude_mask;
    const bits_type b_magnitude = b & magnitude_mask;
    if (a_magnitude > positive_inf || b_magnitude > positive_inf) {
        return false;
    }
    if (a_magnitude == 0 && b_magnitude == 0) {
        return true;
    }
    return a == b;
}

template <DataFormat df, typename number_type>
FORCE_INLINE bool isin_values_equal(number_type a, number_type b) {
    if constexpr (df == DataFormat::Float16_b) {
        return float_bits_equal<uint16_t, 0x7FFF, 0x7F80>(a, b);
    } else if constexpr (df == DataFormat::Float32) {
        uint32_t a_bits;
        uint32_t b_bits;
        __builtin_memcpy(&a_bits, &a, sizeof(a_bits));
        __builtin_memcpy(&b_bits, &b, sizeof(b_bits));
        return float_bits_equal<uint32_t, 0x7FFFFFFF, 0x7F800000>(a_bits, b_bits);
    } else {
        return a == b;
    }
}

constexpr uint32_t ONE_PAGE = 1;
constexpr uint32_t FIRST_STICK = 0;

// Compile-time args 0-2 are unused placeholders. Buffer addresses are runtime args,
// and the empty slots keep TensorAccessorArgs at index 11.
template <
    typename elements_accessor_args_type,
    typename test_elements_accessor_args_type,
    typename output_accessor_args_type>
struct IsInCTAs {
    const uint32_t elements_cb;
    const uint32_t test_elements_cb;
    const uint32_t output_cb;
    const uint32_t elements_size;
    const uint32_t test_elements_size;
    const uint32_t single_fetch_subchunk_size;
    const bool invert;
    const uint32_t elements_tensor_datum_size;
    const uint32_t output_tensor_datum_size;
    const elements_accessor_args_type elements_accessor_args;
    const test_elements_accessor_args_type test_elements_accessor_args;
    const output_accessor_args_type output_accessor_args;
};

// get compile-time arguments by any kernel
FORCE_INLINE constexpr auto get_ctas() {
    constexpr auto elements_args = TensorAccessorArgs<12>();
    constexpr auto test_elements_args = TensorAccessorArgs<elements_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<test_elements_args.next_compile_time_args_offset()>();
    return IsInCTAs<decltype(elements_args), decltype(test_elements_args), decltype(output_args)>{
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9) != 0,
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
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
    Noc noc;
    CircularBuffer cb_obj(cb);
    cb_obj.reserve_back(ONE_PAGE);

    const uint32_t subchunk_size_bytes = subchunk_size * datum_size;
    const uint32_t offset_bytes = offset * datum_size;
    noc.async_read(
        addr_gtor,
        cb_obj,
        subchunk_size_bytes,
        {.page_id = FIRST_STICK, .offset_bytes = offset_bytes},
        {.offset_bytes = 0});
    noc.async_read_barrier();

    cb_obj.push_back(ONE_PAGE);
}

// write from L1 to DRAM
template <typename addr_gen_type>
FORCE_INLINE void write_to_dram(
    const uint32_t& cb,
    const addr_gen_type& addr_gtor,
    const uint32_t& offset,
    const uint32_t& subchunk_size,
    const uint32_t& datum_size) {
    Noc noc;
    CircularBuffer cb_obj(cb);
    cb_obj.wait_front(ONE_PAGE);

    const uint32_t subchunk_size_bytes = subchunk_size * datum_size;
    const uint32_t offset_bytes = offset * datum_size;
    noc.async_write(
        cb_obj,
        addr_gtor,
        subchunk_size_bytes,
        {.offset_bytes = 0},
        {.page_id = FIRST_STICK, .offset_bytes = offset_bytes});
    noc.async_write_barrier();

    cb_obj.pop_front(ONE_PAGE);
}
