// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

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

// Scalar compile-time args. Each kernel builds its DFBs and tensor accessors from its own binding tokens.
struct IsInCTAs {
    const uint32_t elements_size;
    const uint32_t test_elements_size;
    const uint32_t single_fetch_subchunk_size;
    const bool invert;
    const uint32_t elements_tensor_datum_size;
    const uint32_t output_tensor_datum_size;
};

// get compile-time arguments by any kernel
FORCE_INLINE constexpr auto get_ctas() {
    return IsInCTAs{
        get_arg(args::elements_size),
        get_arg(args::test_elements_size),
        get_arg(args::single_fetch_subchunk_size),
        get_arg(args::invert) != 0,
        get_arg(args::elements_tensor_datum_size),
        get_arg(args::output_tensor_datum_size)};
}

// load from DRAM to L1
template <dfb::AccessPattern P, dfb::AccessPattern C, typename addr_gen_type>
FORCE_INLINE void load_to_dfb(
    const DFBBindingToken<P, C>& dfb,
    const addr_gen_type& addr_gtor,
    const uint32_t& offset,
    const uint32_t& subchunk_size,
    const uint32_t& datum_size) {
    Noc noc;
    DataflowBuffer dfb_obj(dfb);
    dfb_obj.reserve_back(ONE_PAGE);

    const uint32_t subchunk_size_bytes = subchunk_size * datum_size;
    const uint32_t offset_bytes = offset * datum_size;
    noc.async_read(
        addr_gtor,
        dfb_obj,
        subchunk_size_bytes,
        {.page_id = FIRST_STICK, .offset_bytes = offset_bytes},
        {.offset_bytes = 0});
    noc.async_read_barrier();

    dfb_obj.push_back(ONE_PAGE);
}

// write from L1 to DRAM
template <dfb::AccessPattern P, dfb::AccessPattern C, typename addr_gen_type>
FORCE_INLINE void write_to_dram(
    const DFBBindingToken<P, C>& dfb,
    const addr_gen_type& addr_gtor,
    const uint32_t& offset,
    const uint32_t& subchunk_size,
    const uint32_t& datum_size) {
    Noc noc;
    DataflowBuffer dfb_obj(dfb);
    dfb_obj.wait_front(ONE_PAGE);

    const uint32_t subchunk_size_bytes = subchunk_size * datum_size;
    const uint32_t offset_bytes = offset * datum_size;
    noc.async_write(
        dfb_obj,
        addr_gtor,
        subchunk_size_bytes,
        {.offset_bytes = 0},
        {.page_id = FIRST_STICK, .offset_bytes = offset_bytes});
    noc.async_write_barrier();

    dfb_obj.pop_front(ONE_PAGE);
}
