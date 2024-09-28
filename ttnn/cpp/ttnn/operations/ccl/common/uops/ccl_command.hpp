// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types.hpp"

namespace ttnn {
namespace ccl {
namespace cmd {

constexpr std::size_t round_up(std::size_t a, std::size_t multiple) {
    return ((a + multiple - 1) / multiple) * multiple;
}

enum class CclCommandArgCode : uint8_t {
    // If operating on a per page granularity
    SET_TENSOR_SHAPE_IN_PAGES = 0,
    SET_TENSOR_SLICE_SHAPE_IN_PAGES,
    SET_TENSOR_SLICE_OFFSET_IN_PAGES,
    SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES,
    SET_WORKER_PAGES_PER_SLICE,
    SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES
};

struct CclCommandTensor {
    Shape4D<uint32_t> tensor_shape;
    Shape4D<uint32_t> tensor_slice_shape;
    Shape4D<uint32_t> tensor_slice_offset;
    Shape4D<uint32_t> worker_start_offset_in_slice;
    uint32_t worker_pages_per_slice;
};

template <CclCommandArgCode code>  struct command_arg_field                                         {  using type = std::nullptr_t; };
template <> struct command_arg_field<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>                  {  using type = Shape4D<uint32_t>; };
template <> struct command_arg_field<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>            {  using type = Shape4D<uint32_t>; };
template <> struct command_arg_field<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>           {  using type = Shape4D<uint32_t>; };
template <> struct command_arg_field<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>  {  using type = Shape4D<uint32_t>; };
template <> struct command_arg_field<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>                 {  using type = uint32_t; };
template <> struct command_arg_field<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>        {  using type = CclCommandTensor; };


template <CclCommandArgCode T>
struct CclCommandArg {

};


using args_elem_t = uint32_t;
template <typename T, CclCommandArgCode CODE>
struct CclCommandArgBase {
    // Let the user override
    using field_type = typename command_arg_field<CODE>::type;  // Ensure T::type is accessible
    static constexpr std::size_t size_in_words() { return (sizeof(T) + sizeof(uint32_t) - 1) / sizeof(uint32_t); }

    static void pack_to(args_elem_t* args, CclCommandTensor const& command_tensor) { T::pack_to(args, command_tensor); }
    static void pack_to(args_elem_t* args, T const& arg) { T::pack_to(args, arg); }
    void pack_to(args_elem_t* args) const { static_cast<T const*>(this)->pack_to(args, this->value); }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) { T::unpack(args, out); }
    static void unpack(volatile args_elem_t const* args, T& out) { T::unpack(args, out); }
    void unpack(volatile args_elem_t const* args) { static_cast<T const*>(this)->unpack(args, &this->value); }

    field_type value;
};

// Note that we choose to reinterpret our pointers as volatile so that in the future we can add streaming
// of additional commands from some backing memory (e.g. dram or L1), potentially by another core, without
// having to track down this code and add volatile casts later (which would be a potentially tricky bug to
// root cause).
inline void unpack_field_without_header(volatile args_elem_t const* args, Shape4D<uint32_t>& out) {
    std::size_t i = 0;
    out.w = args[i++];
    out.z = args[i++];
    out.y = args[i++];
    out.x = args[i++];
}
void pack_field_without_header(args_elem_t* args, Shape4D<uint32_t> const& out);

template <>
struct CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES> : public CclCommandArgBase<CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>, CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES> {

    static void pack_to(args_elem_t* args, CclCommandTensor const& out) {
        pack_field_without_header(&args[0], out.tensor_shape);
    }
    static void pack_to(args_elem_t* args, field_type const& out) { pack_field_without_header(&args[0], out); }
    void pack_to(args_elem_t* args) { pack_field_without_header(&args[0], this->value); }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        unpack_field_without_header(&args[0], out.tensor_shape);
    }
    static void unpack(volatile args_elem_t const* args, field_type& out) { unpack_field_without_header(&args[0], out); }
    void unpack(volatile args_elem_t const* args) { unpack_field_without_header(&args[0], this->value); }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES> : public CclCommandArgBase<CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>, CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES> {

    static void pack_to(args_elem_t* args, CclCommandTensor const& out) {
        pack_field_without_header(&args[0], out.tensor_slice_shape);
    }
    static void pack_to(args_elem_t* args, field_type const& out) { pack_field_without_header(&args[0], out); }
    void pack_to(args_elem_t* args) { pack_field_without_header(&args[0], this->value); }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        unpack_field_without_header(args, out.tensor_slice_shape);
    }
    static void unpack(volatile args_elem_t const* args, field_type& out) { unpack_field_without_header(args, out); }
    void unpack(volatile args_elem_t const* args) { unpack_field_without_header(args, this->value); }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES> : public CclCommandArgBase<CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>, CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES> {
    using type = Shape4D<uint32_t>;

    static void pack_to(args_elem_t* args, CclCommandTensor const& out) {
        pack_field_without_header(args, out.tensor_slice_offset);
    }
    static void pack_to(args_elem_t* args, field_type const& out) { pack_field_without_header(args, out); }
    void pack_to(args_elem_t* args) { pack_field_without_header(args, this->value); }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        unpack_field_without_header(args, out.tensor_slice_offset);
    }
    static void unpack(volatile args_elem_t const* args, field_type& out) { unpack_field_without_header(args, out); }
    void unpack(volatile args_elem_t const* args) { unpack_field_without_header(args, this->value); }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES> : public CclCommandArgBase<CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>, CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES> {
    using type = Shape4D<uint32_t>;

    static void pack_to(args_elem_t* args, CclCommandTensor const& out) {
        pack_field_without_header(args, out.worker_start_offset_in_slice);
    }
    static void pack_to(args_elem_t* args, field_type const& out) { pack_field_without_header(args, out); }
    void pack_to(args_elem_t* args) { pack_field_without_header(args, this->value); }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        unpack_field_without_header(args, out.worker_start_offset_in_slice);
    }
    static void unpack(volatile args_elem_t const* args, field_type& out) { unpack_field_without_header(args, out); }
    void unpack(volatile args_elem_t const* args) { unpack_field_without_header(args, this->value); }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE> : public CclCommandArgBase<CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>, CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE> {
    using type = uint32_t;

    static void pack_to(args_elem_t* args, CclCommandTensor const& out) { args[0] = out.worker_pages_per_slice; }
    static void pack_to(args_elem_t* args, field_type const& out) { args[0] = out; }
    void pack_to(args_elem_t* args) const { args[0] = this->value; }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) { out.worker_pages_per_slice = args[0]; }
    static void unpack(volatile args_elem_t const* args, field_type& out) { out = args[0]; }
    void unpack(volatile args_elem_t const* args) { this->value = args[0]; }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>
    : public CclCommandArgBase<CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>, CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES> {
    using type = CclCommandTensor;

    // considering making this some generator type that implements operator[]
    // so I can have tests that make sure I don't go OoB so I can make sure `size_in_words`
    // is correct
    static void pack_to(args_elem_t* args, field_type const& command_tensor) {
        std::size_t i = 0;

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::pack_to(&args[i], command_tensor.tensor_shape);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::pack_to(&args[i], command_tensor.tensor_slice_shape);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::pack_to(&args[i], command_tensor.tensor_slice_offset);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::pack_to(&args[i], command_tensor.worker_start_offset_in_slice);
        i += CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::pack_to(&args[i], command_tensor.worker_pages_per_slice);
        i += CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::size_in_words();
    }

    void pack_to(args_elem_t* args) const {
        CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::pack_to(args, this->value);
    }

    // TODO: when kernels get c++20, use std::span
    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        std::size_t i = 0;
        CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::unpack(&args[i], out.tensor_shape);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::unpack(&args[i], out.tensor_slice_shape);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::unpack(&args[i], out.tensor_slice_offset);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::unpack(&args[i], out.worker_start_offset_in_slice);
        i += CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::unpack(&args[i], out.worker_pages_per_slice);
        i += CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::size_in_words();
    }

    void unpack(volatile args_elem_t const* args) {
        CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::unpack(args, this->value);
    }
};


// Convenience type aliases
using tensor_shape_command_arg_t = CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>;
using tensor_slice_shape_command_arg_t = CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>;
using tensor_slice_offset_command_arg_t = CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>;
using worker_start_offset_command_arg_t = CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>;
using worker_pages_command_arg_t = CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>;
using full_tensor_command_arg_t = CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>;

// A command is composed of one or more arguments
// This enum specifies the high level command
// Future commands are to be added and will enable
// functionalilty such as synchronizing
enum class CclCommandCode : uint8_t {
    STREAM_TENSOR_TO_EDM = 0,
    STREAM_EDM_TO_TENSOR
};

struct CclCommandHeader {
    CclCommandCode code;

    // For the time being we have a dedicated arg_count because we assume
    // we may save args/tensor info from previous command. Up to command sequence
    // generator to make sure any fields/args not explicitly listed are correct from prior command
    uint8_t arg_count : 4;
    uint8_t reserved1;
    uint8_t reserved2;

    static CclCommandHeader from_uint32(uint32_t const& cmd_header) {
        CclCommandHeader decoded;
        decoded.code = static_cast<CclCommandCode>(cmd_header & 0xFF);
        decoded.arg_count = (cmd_header >> 8) & 0xF;
        return decoded;
    }

    static uint32_t to_uint32(CclCommandHeader const& cmd_header) {
        uint32_t encoded = 0;
        encoded = (uint8_t)(cmd_header.code);
        encoded = encoded | (cmd_header.arg_count << 8);
        return encoded;
    }
    uint32_t to_uint32() const {
        return *reinterpret_cast<uint32_t const*>(this);
    }
};
static_assert(sizeof(CclCommandHeader) == sizeof(uint32_t));


}  // namespace cmd
}  // namespace ccl
}  // namespace ttnn
