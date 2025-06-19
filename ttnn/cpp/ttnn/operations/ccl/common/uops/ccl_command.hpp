// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <cstddef>
#include <cstdint>
#include <variant>
#include <limits>

#include "ttnn/operations/ccl/common/types/ccl_types.hpp"
// For command dest type
#include <tt-metalium/fabric_edm_packet_header.hpp>

namespace ttnn {
namespace ccl {
namespace v2 {
struct TensorSlice {
    using ords_t = Shape4D<uint32_t>;
    ords_t tensor_shape;
    ords_t tensor_slice_shape;
    ords_t tensor_slice_offset;
    ords_t worker_slice_shape;
    ords_t worker_slice_offset;

    TensorSlice(TensorSlice const& rhs) = default;
    TensorSlice(TensorSlice&& rhs) = default;
    TensorSlice& operator=(TensorSlice const& rhs) = default;
    TensorSlice& operator=(TensorSlice&& rhs) = default;

    TensorSlice() = default;
    TensorSlice(
        ords_t tensor_shape,
        ords_t tensor_slice_shape,
        ords_t tensor_slice_offset,
        ords_t worker_slice_shape,
        ords_t worker_slice_offset) :
        tensor_shape(tensor_shape),
        tensor_slice_shape(tensor_slice_shape),
        tensor_slice_offset(tensor_slice_offset),
        worker_slice_shape(worker_slice_shape),
        worker_slice_offset(worker_slice_offset) {}
};
}  // namespace v2

namespace cmd {

constexpr std::size_t round_up(std::size_t a, std::size_t multiple) {
    return ((a + multiple - 1) / multiple) * multiple;
}

// for CclCommandStreamTensorToCB and CclCommandStreamCBToTensor
using CclCommandStreamTensorSlice = v2::TensorSlice;
struct CclCommandWaitValue {
    uint32_t target_value = 0;
};
struct CclCommandAtomicInc {
    uint32_t value = 1;
    uint32_t wrap_value = std::numeric_limits<uint32_t>::max();
};

struct noc_transfer_info {
    // When encoded into a command, the noc address contains the relative offset of the
    // read/write from the base address, which must be stored elsewhere by the command
    // interpreter/command stream. The base address is specified in the command
    uint64_t noc_addr = 0;
    size_t noc_transfer_size_bytes = 0;
};

struct HostNocTransferBurstGrouping {
    size_t num_transfers_per_packet = 0;
    std::vector<noc_transfer_info> transfer_infos;
};
struct HostCclCommandNocTransferBurst {
    size_t bank_base_address = 0;
    uint32_t num_transfers_total = 0;
    std::vector<HostNocTransferBurstGrouping> transfer_burst_groupings;
};
struct DeviceCclCommandNocTransferBurst {
    size_t bank_base_address = 0;
    uint32_t num_transfers_total = 0;

    // Populated as the burst is being completed and command args are being decoded
    uint8_t num_transfers_per_packet = 0;
};

struct CclCommandInlineReadWrite {
    uint32_t value = 0;
};
struct CclCommandReadWrite {
    uint32_t size_bytes = 0;
};
using CclCommandArgs = std::variant<
    CclCommandStreamTensorSlice,
    CclCommandWaitValue,
    CclCommandAtomicInc,
    CclCommandInlineReadWrite,
    CclCommandReadWrite,
    HostCclCommandNocTransferBurst>;

enum SRC_DEST_TYPE : uint8_t { SRC = 0, DEST = 1 };

// Explicitly assigned integer values for easier debug
enum class CclCommandArgCode : uint8_t {
    // If operating on a per page granularity
    SET_TENSOR_SHAPE_IN_PAGES = 0,
    SET_TENSOR_SLICE_SHAPE_IN_PAGES = 1,
    SET_TENSOR_SLICE_OFFSET_IN_PAGES = 2,
    SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES = 3,
    SET_WORKER_PAGES_PER_SLICE = 4,
    SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES = 5,

    // wait_value, inline read/write
    SET_TARGET_VALUE = 6,
    SET_ATOMIC_INC_VALUE = 7,

    // addr type commands
    SET_ADDRESS_INFO = 8,

    // core descriptor commands
    SET_CORE_DESCRIPTOR_INFO = 9,

    // Specifies how many noc transfers are specified in the
    // noc transaction burst command
    SET_NOC_TRANSFER_BURST_START_INFO = 10,

    // Specifies how many noc transfers are expected to be performed back to back
    // and packed into a single (CB) packet (though conceivably we could pack this)
    // into a common ethernet packet too for better utilization (provided that the)
    // receiver does the proper unpacking
    SET_NOC_TRANSFER_BURST_SIZE_PER_PACKET = 11,

    INVALID = std::numeric_limits<uint8_t>::max(),
};

struct CclCommandArgHeader {
    CclCommandArgCode code = CclCommandArgCode::INVALID;
    uint8_t inline_value0 = 0;
    uint8_t inline_value1 = 0;
    uint8_t inline_value2 = 0;

    static CclCommandArgHeader from_uint32(uint32_t val) {
        CclCommandArgHeader header;
        header.code = static_cast<CclCommandArgCode>(val & 0xFF);
        header.inline_value0 = (val >> 8) & 0xFF;
        header.inline_value1 = (val >> 16) & 0xFF;
        header.inline_value2 = (val >> 24) & 0xFF;
        return header;
    }
    uint32_t to_uint32() const {
        uint32_t val = 0;
        val |= static_cast<uint32_t>(this->code);
        val |= static_cast<uint32_t>(this->inline_value0) << 8;
        val |= static_cast<uint32_t>(this->inline_value1) << 16;
        val |= static_cast<uint32_t>(this->inline_value2) << 24;
        return val;
    }
};
static_assert(sizeof(CclCommandArgHeader) == sizeof(uint32_t));

struct CclCommandTensor {
    Shape4D<uint32_t> tensor_shape;
    Shape4D<uint32_t> tensor_slice_shape;
    Shape4D<uint32_t> tensor_slice_offset;
    Shape4D<uint32_t> worker_start_offset_in_slice;
    uint32_t worker_pages_per_slice;
};

template <CclCommandArgCode code>
struct command_arg_field {
    using type = std::nullptr_t;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES> {
    using type = Shape4D<uint32_t>;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES> {
    using type = Shape4D<uint32_t>;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES> {
    using type = Shape4D<uint32_t>;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES> {
    using type = Shape4D<uint32_t>;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE> {
    using type = uint32_t;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_TARGET_VALUE> {
    using type = uint32_t;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_ATOMIC_INC_VALUE> {
    using type = CclCommandAtomicInc;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES> {
    using type = CclCommandTensor;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_ADDRESS_INFO> {
    using type = uint32_t;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_NOC_TRANSFER_BURST_START_INFO> {
    using type = DeviceCclCommandNocTransferBurst;
};
template <>
struct command_arg_field<CclCommandArgCode::SET_NOC_TRANSFER_BURST_SIZE_PER_PACKET> {
    using type = uint8_t;
};

template <CclCommandArgCode T>
struct CclCommandArg {};

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
struct CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>
    : public CclCommandArgBase<
          CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>,
          CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES> {
    static void pack_to(args_elem_t* args, CclCommandTensor const& out) {
        pack_field_without_header(&args[0], out.tensor_shape);
    }
    static void pack_to(args_elem_t* args, field_type const& out) { pack_field_without_header(&args[0], out); }
    void pack_to(args_elem_t* args) { pack_field_without_header(&args[0], this->value); }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        unpack_field_without_header(&args[0], out.tensor_shape);
    }
    static void unpack(volatile args_elem_t const* args, field_type& out) {
        unpack_field_without_header(&args[0], out);
    }
    void unpack(volatile args_elem_t const* args) { unpack_field_without_header(&args[0], this->value); }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>
    : public CclCommandArgBase<
          CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>,
          CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES> {
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
struct CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>
    : public CclCommandArgBase<
          CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>,
          CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES> {
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
struct CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>
    : public CclCommandArgBase<
          CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>,
          CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES> {
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
struct CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>
    : public CclCommandArgBase<
          CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>,
          CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE> {
    using type = uint32_t;

    static void pack_to(args_elem_t* args, CclCommandTensor const& out) { args[0] = out.worker_pages_per_slice; }
    static void pack_to(args_elem_t* args, field_type const& out) { args[0] = out; }
    void pack_to(args_elem_t* args) const { args[0] = this->value; }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        out.worker_pages_per_slice = args[0];
    }
    static void unpack(volatile args_elem_t const* args, field_type& out) { out = args[0]; }
    void unpack(volatile args_elem_t const* args) { this->value = args[0]; }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>
    : public CclCommandArgBase<
          CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>,
          CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES> {
    using type = CclCommandTensor;

    // considering making this some generator type that implements operator[]
    // so I can have tests that make sure I don't go OoB so I can make sure `size_in_words`
    // is correct
    static void pack_to(args_elem_t* args, field_type const& command_tensor) {
        std::size_t i = 0;

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::pack_to(&args[i], command_tensor.tensor_shape);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::pack_to(
            &args[i], command_tensor.tensor_slice_shape);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::pack_to(
            &args[i], command_tensor.tensor_slice_offset);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::pack_to(
            &args[i], command_tensor.worker_start_offset_in_slice);
        i += CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::pack_to(
            &args[i], command_tensor.worker_pages_per_slice);
        i += CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::size_in_words();
    }

    void pack_to(args_elem_t* args) const {
        CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::pack_to(args, this->value);
    }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        std::size_t i = 0;
        CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::unpack(&args[i], out.tensor_shape);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::unpack(&args[i], out.tensor_slice_shape);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::unpack(&args[i], out.tensor_slice_offset);
        i += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::unpack(
            &args[i], out.worker_start_offset_in_slice);
        i += CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::size_in_words();

        CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::unpack(&args[i], out.worker_pages_per_slice);
        i += CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::size_in_words();
    }

    void unpack(volatile args_elem_t const* args) {
        CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::unpack(args, this->value);
    }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_TARGET_VALUE> : public CclCommandArgBase<
                                                                CclCommandArg<CclCommandArgCode::SET_TARGET_VALUE>,
                                                                CclCommandArgCode::SET_TARGET_VALUE> {
    static void pack_to(args_elem_t* args, uint32_t value) { args[0] = value; }
    void pack_to(args_elem_t* args) { pack_to(&args[0], this->value); }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        unpack_field_without_header(&args[0], out.tensor_shape);
    }
    static void unpack(volatile args_elem_t const* args, field_type& out) { out = args[0]; }
    void unpack(volatile args_elem_t const* args) { this->value = args[0]; }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_ATOMIC_INC_VALUE>
    : public CclCommandArgBase<
          CclCommandArg<CclCommandArgCode::SET_ATOMIC_INC_VALUE>,
          CclCommandArgCode::SET_ATOMIC_INC_VALUE> {
    static void pack_to(args_elem_t* args, CclCommandAtomicInc const& atomic_inc_args) {
        args[0] = atomic_inc_args.value;
        args[1] = atomic_inc_args.wrap_value;
    }
    void pack_to(args_elem_t* args) { pack_to(&args[0], this->value); }

    static void unpack(volatile args_elem_t const* args, CclCommandAtomicInc& out) {
        out.value = args[0];
        out.wrap_value = args[1];
    }
    void unpack(volatile args_elem_t const* args) {
        this->value.value = args[0];
        this->value.wrap_value = args[1];
    }
};

template <>
struct CclCommandArg<CclCommandArgCode::SET_ADDRESS_INFO> : public CclCommandArgBase<
                                                                CclCommandArg<CclCommandArgCode::SET_ADDRESS_INFO>,
                                                                CclCommandArgCode::SET_ADDRESS_INFO> {
    static void pack_to(args_elem_t* args, uint32_t value) { args[0] = value; }
    void pack_to(args_elem_t* args) { pack_to(&args[0], this->value); }

    static void unpack(volatile args_elem_t const* args, CclCommandTensor& out) {
        unpack_field_without_header(&args[0], out.tensor_shape);
    }
    static void unpack(volatile args_elem_t const* args, field_type& out) { out = args[0]; }
    void unpack(volatile args_elem_t const* args) { this->value = args[0]; }
};

// Convenience type aliases
using tensor_shape_command_arg_t = CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>;
using tensor_slice_shape_command_arg_t = CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>;
using tensor_slice_offset_command_arg_t = CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>;
using worker_start_offset_command_arg_t = CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>;
using worker_pages_command_arg_t = CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>;
using full_tensor_command_arg_t = CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>;

enum class CclCommandAddrType : uint8_t {
    SEMAPHORE_ID,
    CIRCULAR_BUFFER_ID,
    ABSOLUTE_ADDRESS,
    RELATIVE_ADDRESS,

    // Useful for inline commands (read/write, atomic inc)
    NONE
};
struct CclCommandAddrSemaphoreId {
    uint32_t semaphore_id;
};
struct CclCommandAddrCircularBufferId {
    uint32_t circular_buffer_id;
};
struct CclCommandAddrAbsoluteAddress {
    uint32_t absolute_address;
};
struct CclCommandAddrRelativeAddress {
    uint32_t relative_address;
};
struct CclCommandAddrNone {};

using CclCommandAddrArgs = std::variant<
    CclCommandAddrSemaphoreId,
    CclCommandAddrCircularBufferId,
    CclCommandAddrAbsoluteAddress,
    CclCommandAddrRelativeAddress,
    CclCommandAddrNone>;

enum class CclCommandCoreDescriptorType : uint8_t {
    // Temporary since at the moment, tensor commands have their source/dest type implied
    // by the command stream index - the info is all off the addrgen
    ADDRGEN = 0,
    LOCAL = 1,
    NOC_XY = 2,
    RECTANGLE = 3,

    // used for noc bursts since the core is embedded in the noc command
    NONE = 4
    // Future types may include: list, rectangle_list, etc.
};
struct CclCommandCoreDescriptorTypeAddrgen {};
struct CclCommandCoreDescriptorTypeNone {};
struct CclCommandCoreDescriptorTypeLocal {};
struct CclCommandCoreDescriptorTypeNocXY {
    uint8_t x;
    uint8_t y;
};
// unused atm
struct CclCommandCoreDescriptorTypeMcast {
    uint32_t to_uint32() const {
        uint32_t value = 0;
        value |= (noc0_start_x << 0);
        value |= (noc0_start_y << 8);
        value |= (noc0_end_x << 16);
        value |= (noc0_end_y << 24);
        return value;
    }
    static CclCommandCoreDescriptorTypeMcast from_uint32(uint32_t value) {
        CclCommandCoreDescriptorTypeMcast mcast;
        mcast.noc0_start_x = (value >> 0) & 0xFF;
        mcast.noc0_start_y = (value >> 8) & 0xFF;
        mcast.noc0_end_x = (value >> 16) & 0xFF;
        mcast.noc0_end_y = (value >> 24) & 0xFF;
        return mcast;
    }

    uint8_t noc0_start_x;
    uint8_t noc0_start_y;
    uint8_t noc0_end_x;
    uint8_t noc0_end_y;
};
using CclCommandCoreDescriptorArgs = std::variant<
    CclCommandCoreDescriptorTypeAddrgen,
    CclCommandCoreDescriptorTypeLocal,
    CclCommandCoreDescriptorTypeNocXY,
    CclCommandCoreDescriptorTypeMcast,
    CclCommandCoreDescriptorTypeNone>;

// A command is composed of one or more arguments
// This enum specifies the high level command
// Future commands are to be added and will enable
// functionalilty such as synchronizing
enum class CclCommandCode : uint8_t {
    STREAM_TENSOR_TO_EDM = 0,  // TODO: rename uses of to the below
    STREAM_TENSOR_TO_CB = 0,
    STREAM_CB_TO_TENSOR = 1,
    STREAM_EDM_TO_TENSOR = 2,  // TODO: rename uses of to the above

    WAIT_VALUE = 3,

    // value, wrap, dest_type, dest_addr_info
    ATOMIC_INC = 4,

    RAW_INLINE_WRITE_BYTES = 5,

    NOC_READ_BURST = 6,
    NOC_WRITE_BURST = 7,

    // Waits on semaphore values before performing reads. Every read waits for the target semaphore
    // value before reading
    FLOW_CONTROLLED_NOC_READ_BURST = 8,
    NOC_WRITE_AND_ATOMIC_INC = 9,

    INVALID = 10
};


enum CclCommandDestType : uint8_t {
    CHIP_UNICAST = tt::tt_fabric::CHIP_UNICAST,
    CHIP_MULTICAST = tt::tt_fabric::CHIP_MULTICAST,
    CHIP_LOCAL_ONLY = 2
};
static_assert(tt::tt_fabric::CHIP_UNICAST < 2);
static_assert(tt::tt_fabric::CHIP_MULTICAST < 2);
struct DestTypeArgsNull {};
static_assert(sizeof(DestTypeArgsNull) <= 2);
struct UnicastCommandDestArgs {
    uint8_t distance_in_hops;
    bool is_forward_direction;
};
struct MulticastCommandDestArgs {
    uint8_t num_targets_forward_direction;
    uint8_t num_targets_backward_direction;
};
using LocalOnlyCommandDestArgs = DestTypeArgsNull;

// Used only for host code paths
using CclCommandDestArgs = std::variant<UnicastCommandDestArgs, MulticastCommandDestArgs, LocalOnlyCommandDestArgs>;


struct CclCommandHeader {
    CclCommandCode code : 6;
    CclCommandDestType dest_type : 2;

    // For the time being we have a dedicated arg_count because we assume
    // we may save args/tensor info from previous command. Up to command sequence
    // generator to make sure any fields/args not explicitly listed are correct from prior command
    uint8_t arg_count : 4;
    union {
        DestTypeArgsNull null;
        UnicastCommandDestArgs unicast;
        MulticastCommandDestArgs multicast;
        LocalOnlyCommandDestArgs local_only;
    } command_dest_args;

    CclCommandHeader() :
        code(CclCommandCode::INVALID), dest_type(CclCommandDestType::CHIP_LOCAL_ONLY), arg_count(0) {}
    CclCommandHeader(CclCommandCode code, const CclCommandDestArgs& args, uint8_t arg_count) :
        code(code), arg_count(arg_count) {
        if (std::holds_alternative<UnicastCommandDestArgs>(args)) {
            command_dest_args.unicast = std::get<UnicastCommandDestArgs>(args);
            this->dest_type = CclCommandDestType::CHIP_UNICAST;
        } else if (std::holds_alternative<MulticastCommandDestArgs>(args)) {
            command_dest_args.multicast = std::get<MulticastCommandDestArgs>(args);
            this->dest_type = CclCommandDestType::CHIP_MULTICAST;
        } else if (std::holds_alternative<LocalOnlyCommandDestArgs>(args)) {
            command_dest_args.local_only = std::get<LocalOnlyCommandDestArgs>(args);
            this->dest_type = CclCommandDestType::CHIP_LOCAL_ONLY;
        }
    }
    CclCommandHeader(CclCommandCode code, const MulticastCommandDestArgs& multicast_args, uint8_t arg_count) :
        code(code), dest_type(CclCommandDestType::CHIP_MULTICAST), arg_count(arg_count) {
        this->command_dest_args.multicast = multicast_args;
    }
    CclCommandHeader(CclCommandCode code, const LocalOnlyCommandDestArgs& local_only_args, uint8_t arg_count) :
        code(code), dest_type(CclCommandDestType::CHIP_LOCAL_ONLY), arg_count(arg_count) {
        this->command_dest_args.local_only = local_only_args;
    }

    static CclCommandHeader from_uint32_impl(uint32_t cmd_header) {
        CclCommandHeader decoded;
        reinterpret_cast<uint32_t*>(&decoded)[0] = cmd_header;
        return decoded;
    }

    static uint32_t to_uint32(const CclCommandHeader& cmd_header) {
        uint32_t encoded = 0;
        encoded = (uint32_t)(cmd_header.code);
        encoded |= (cmd_header.dest_type << 6);
        encoded |= (cmd_header.arg_count << 8);
        switch (cmd_header.dest_type) {
            case CclCommandDestType::CHIP_UNICAST:
                encoded |= (cmd_header.command_dest_args.unicast.distance_in_hops << 16);
                encoded |= (cmd_header.command_dest_args.unicast.is_forward_direction << 24);
                break;
            case CclCommandDestType::CHIP_MULTICAST:
                encoded |= (cmd_header.command_dest_args.multicast.num_targets_forward_direction << 16);
                encoded |= (cmd_header.command_dest_args.multicast.num_targets_backward_direction << 24);
                break;
            default: break;
        };
        return encoded;
    }
    uint32_t to_uint32() const { return to_uint32(*this); }
    static CclCommandHeader from_uint32(uint32_t cmd_header) {
        return CclCommandHeader::from_uint32_impl(cmd_header);
    }

    const UnicastCommandDestArgs& get_unicast_dest_args() const { return command_dest_args.unicast; }
    const MulticastCommandDestArgs& get_multicast_dest_args() const { return command_dest_args.multicast; }
    const LocalOnlyCommandDestArgs& get_local_only_dest_args() const { return command_dest_args.local_only; }
};

static_assert(sizeof(CclCommandHeader) == sizeof(uint32_t));

}  // namespace cmd
}  // namespace ccl
}  // namespace ttnn
