// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "cfg.h"
#include "ckernel.h"
#include "dev_mem_map.h"
#include "sync.h"
#include "utils/gpr.h"

namespace hal::move
{

struct SrcA
{
};

struct SrcB
{
};

struct Dst
{
};

struct L1
{
};

struct Gpr
{
};

struct Mmio
{
};

struct Zero
{
};

struct BackendConfig
{
};

struct NcriscIram
{
};

/** @brief Number of destination rows produced by a matrix-register transfer. */
struct RowCount
{
    std::uint16_t value;
};

namespace rows
{

inline constexpr RowCount One {1};
inline constexpr RowCount Four {4};
inline constexpr RowCount Eight {8};
inline constexpr RowCount Face {16};

/** @brief Construct a positive whole-face row count. */
template <std::uint16_t Count>
constexpr RowCount faces()
{
    static_assert(Count > 0, "face count must be positive");
    static_assert(Count <= 0xffffu / 16u, "row count does not fit RowCount");

    return RowCount {static_cast<std::uint16_t>(Count * 16u)};
}

} // namespace rows

/** @brief Return whether a row count is one, four, eight, or a positive whole number of faces. */
constexpr bool is_canonical(const RowCount rows)
{
    return rows.value == 1u || rows.value == 4u || rows.value == 8u || (rows.value != 0u && rows.value % 16u == 0u);
}

/**
 * @brief Constexpr Dst row geometry shared by face- and tile-addressed matrix transfers.
 */
namespace dst_layout
{

inline constexpr std::uint32_t RowsPerFace  = 16u;
inline constexpr std::uint32_t FacesPerTile = 4u;
inline constexpr std::uint32_t RowsPerTile  = FacesPerTile * RowsPerFace;

/** @brief Return the Dst row of one face row inside one 32x32 tile. */
constexpr std::uint32_t tile_row(const std::uint32_t tile_index, const std::uint32_t face = 0u, const std::uint32_t row = 0u)
{
    return tile_index * RowsPerTile + face * RowsPerFace + row;
}

} // namespace dst_layout

/** @brief Select SrcB replication performed by MOVB2D. */
enum class Broadcast : std::uint8_t
{
    None,
    Column0,
    Row,
    Scalar
};

/** @brief Select whether a source-to-Dst move honors the source-bank client wait. */
enum class SourceValidity : std::uint8_t
{
    Wait,
    Ignore
};

/** @brief Select whether a ten-bit Dst row field holds an absolute row or a masked signed counter offset. */
enum class RowAddressing : std::uint8_t
{
    Absolute,
    CounterRelative
};

/** @brief Reuse address_mode for the final generated instruction. */
inline constexpr std::uint8_t SameAddressMode = 0xffu;

/** @brief Maximum number of native TT instructions emitted by one matrix transfer. */
inline constexpr std::size_t MaxMatrixTransferInstructions = 8u;

/** @brief Select the amount transferred by LOADIND or STOREIND-to-L1. */
enum class ScalarSize : std::uint8_t
{
    Bytes16,
    Bytes4,
    Bytes2,
    Bytes1
};

/** @brief Select the LOADIND or STOREIND offset-register side effect. */
enum class OffsetIncrement : std::uint8_t
{
    None,
    Bytes2,
    Bytes4,
    Bytes16
};

/** @brief Select whether run() issues only or also installs a completion wait. */
enum class Completion : std::uint8_t
{
    IssueOnly,
    Wait
};

namespace detail
{

/** @brief Normalize an existing hal::Gpr operand for structural descriptor storage. */
struct GprRef
{
    std::uint32_t index = 0xffffffffu;

    constexpr GprRef() = default;

    template <std::uint32_t Index>
    constexpr GprRef(const hal::Gpr<Index>) : index(Index)
    {
    }

    constexpr GprRef(const hal::Gpr<hal::detail::DynamicGprIndex> gpr) : index(gpr.index)
    {
    }
};

} // namespace detail

/** @brief Select one 16-bit half of a Tensix GPR. */
enum class GprHalf : std::uint8_t
{
    Low,
    High
};

/** @brief Identify one of the current thread's 128 GPR half-registers. */
struct GprHalfRef
{
    detail::GprRef gpr {};
    GprHalf half = GprHalf::Low;
};

/** @brief Address L1 using the GPR pair consumed by LOADIND and STOREIND. */
struct IndirectL1Address
{
    detail::GprRef base {};
    GprHalfRef offset {};
    OffsetIncrement offset_increment = OffsetIncrement::None;
};

/** @brief Address SrcA or SrcB using STOREIND's encoded source address. */
struct IndirectSrcAddress
{
    detail::GprRef base {};
    GprHalfRef offset {};
    OffsetIncrement offset_increment = OffsetIncrement::None;
};

/** @brief Select the MMIO address formation and therefore the write opcode. */
enum class MmioAddressing : std::uint8_t
{
    Invalid,
    Immediate,
    Indirect
};

/** @brief Describe an immediate or GPR-indirect address in the MMIO space. */
struct MmioAddress
{
    MmioAddressing addressing  = MmioAddressing::Invalid;
    std::uint32_t byte_address = 0;
    detail::GprRef base {};
    GprHalfRef offset {};
    OffsetIncrement offset_increment = OffsetIncrement::None;
};

/** @brief Identify an L1 address in XMOV's 16-byte blocks. */
struct L1Address16B
{
    std::uint32_t block = 0xffffffffu;
};

/** @brief Identify a backend-CFG offset in XMOV's 16-byte blocks. */
struct BackendConfigAddress16B
{
    std::uint32_t block_offset = 0xffffffffu;
};

/** @brief Identify an NCRISC-IRAM offset in XMOV's 16-byte blocks. */
struct NcriscIramAddress16B
{
    std::uint32_t block_offset = 0xffffffffu;
};

/** @brief Identify an XMOV transfer length in 16-byte blocks. */
struct BlockCount16B
{
    std::uint32_t value = 0;
};

// Unsupported source/destination pairs have no definition.
template <typename Source, typename Destination>
struct Transfer;

/** @brief Describe a MOVA2D or MOVDBGA2D transfer. */
template <>
struct Transfer<SrcA, Dst>
{
    std::uint32_t source_row             = 0xffffffffu;
    std::uint32_t destination_row        = 0xffffffffu;
    RowAddressing destination_addressing = RowAddressing::Absolute;
    std::uint8_t address_mode            = 0xffu;
    RowCount number_of_rows {0};
    std::uint8_t final_address_mode = SameAddressMode;
    bool destination_32bit_low      = false;
    SourceValidity source_validity  = SourceValidity::Wait;

    /** @brief Encode the transfer when it consists of exactly one instruction. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Describe a MOVB2D or MOVDBGB2D transfer. */
template <>
struct Transfer<SrcB, Dst>
{
    std::uint32_t source_row             = 0xffffffffu;
    std::uint32_t destination_row        = 0xffffffffu;
    RowAddressing destination_addressing = RowAddressing::Absolute;
    std::uint8_t address_mode            = 0xffu;
    RowCount number_of_rows {0};
    Broadcast broadcast             = Broadcast::None;
    std::uint8_t final_address_mode = SameAddressMode;
    bool destination_32bit_low      = false;
    SourceValidity source_validity  = SourceValidity::Wait;

    /** @brief Encode the transfer when it consists of exactly one instruction. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Describe a MOVD2B transfer. */
template <>
struct Transfer<Dst, SrcB>
{
    std::uint32_t source_row        = 0xffffffffu;
    RowAddressing source_addressing = RowAddressing::Absolute;
    std::uint32_t destination_row   = 0xffffffffu;
    std::uint8_t address_mode       = 0xffu;
    RowCount number_of_rows {0};
    std::uint8_t final_address_mode = SameAddressMode;
    bool source_32bit_low           = false;

    /** @brief Encode the transfer when it consists of exactly one instruction. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Describe an asynchronous LOADIND transfer from L1 to GPRs. */
template <>
struct Transfer<L1, Gpr>
{
    IndirectL1Address source {};
    detail::GprRef destination {};
    ScalarSize size       = ScalarSize::Bytes16;
    Completion completion = Completion::IssueOnly;

    /** @brief Encode the transfer when completion does not append a wait. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Describe a posted STOREIND transfer from GPRs to L1. */
template <>
struct Transfer<Gpr, L1>
{
    detail::GprRef source {};
    IndirectL1Address destination {};
    ScalarSize size       = ScalarSize::Bytes16;
    Completion completion = Completion::IssueOnly;

    /** @brief Encode the transfer when completion does not append a wait. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Describe a fixed-width STOREIND transfer from GPRs to SrcA. */
template <>
struct Transfer<Gpr, SrcA>
{
    detail::GprRef source {};
    IndirectSrcAddress destination {};

    /** @brief Encode the one-instruction transfer. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Describe a fixed-width STOREIND transfer from GPRs to SrcB. */
template <>
struct Transfer<Gpr, SrcB>
{
    detail::GprRef source {};
    IndirectSrcAddress destination {};

    /** @brief Encode the one-instruction transfer. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Describe an immediate STOREREG or indirect STOREIND MMIO write. */
template <>
struct Transfer<Gpr, Mmio>
{
    detail::GprRef source {};
    MmioAddress destination {};
    Completion completion = Completion::IssueOnly;

    /** @brief Encode the transfer when completion does not append a wait. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Describe an asynchronous direct LOADREG MMIO read. */
template <>
struct Transfer<Mmio, Gpr>
{
    MmioAddress source {};
    detail::GprRef destination {};
    Completion completion = Completion::IssueOnly;

    /** @brief Encode the transfer when completion does not append a wait. */
    constexpr std::uint32_t get_operation() const;
};

/** @brief Describe Blackhole's supported L1-to-L1 XMOV composite. */
template <>
struct Transfer<L1, L1>
{
    L1Address16B source {};
    L1Address16B destination {};
    BlockCount16B size {};
    Completion completion = Completion::IssueOnly;

    constexpr std::uint32_t get_operation() const = delete;
};

namespace detail
{

inline constexpr std::uint32_t MatrixSourceRows      = 64u;
inline constexpr std::uint32_t MatrixDestinationRows = 1024u;
inline constexpr std::uint32_t AddressModeCount      = 8u;
inline constexpr std::uint32_t MmioBase              = 0xffb00000u;
inline constexpr std::uint32_t MmioFirstEncodable    = 0xffb11000u;
inline constexpr std::uint32_t MmioLastEncodable     = 0xffbffffcu;
inline constexpr std::uint32_t L1BlockCapacity       = MEM_L1_SIZE / 16u;

inline constexpr std::uint32_t LoadIndOpcode    = 0x49u;
inline constexpr std::uint32_t MoveDbgA2DOpcode = 0x09u;
inline constexpr std::uint32_t MoveD2BOpcode    = 0x0au;
inline constexpr std::uint32_t MoveDbgB2DOpcode = 0x0cu;
inline constexpr std::uint32_t MoveA2DOpcode    = 0x12u;
inline constexpr std::uint32_t MoveB2DOpcode    = 0x13u;
inline constexpr std::uint32_t XmovOpcode       = 0x40u;
inline constexpr std::uint32_t StoreIndOpcode   = 0x66u;
inline constexpr std::uint32_t StoreRegOpcode   = 0x67u;
inline constexpr std::uint32_t LoadRegOpcode    = 0x68u;

constexpr std::uint32_t encode_instruction(const std::uint32_t opcode, const std::uint32_t fields)
{
    return (opcode << 24u) | fields;
}

constexpr bool is_valid(const Broadcast broadcast)
{
    return broadcast == Broadcast::None || broadcast == Broadcast::Column0 || broadcast == Broadcast::Row || broadcast == Broadcast::Scalar;
}

constexpr bool is_valid(const SourceValidity validity)
{
    return validity == SourceValidity::Wait || validity == SourceValidity::Ignore;
}

constexpr bool is_valid(const ScalarSize size)
{
    return size == ScalarSize::Bytes16 || size == ScalarSize::Bytes4 || size == ScalarSize::Bytes2 || size == ScalarSize::Bytes1;
}

constexpr bool is_valid(const OffsetIncrement increment)
{
    return increment == OffsetIncrement::None || increment == OffsetIncrement::Bytes2 || increment == OffsetIncrement::Bytes4 ||
           increment == OffsetIncrement::Bytes16;
}

constexpr bool is_valid(const Completion completion)
{
    return completion == Completion::IssueOnly || completion == Completion::Wait;
}

constexpr bool is_valid(const GprHalf half)
{
    return half == GprHalf::Low || half == GprHalf::High;
}

constexpr bool is_valid(const MmioAddressing addressing)
{
    return addressing == MmioAddressing::Immediate || addressing == MmioAddressing::Indirect;
}

constexpr bool is_valid(const GprRef gpr)
{
    return gpr.index < 64u;
}

constexpr bool is_valid(const GprHalfRef half)
{
    return is_valid(half.gpr) && is_valid(half.half);
}

constexpr bool is_valid(const IndirectL1Address address)
{
    return is_valid(address.base) && is_valid(address.offset) && is_valid(address.offset_increment);
}

constexpr bool is_valid(const IndirectSrcAddress address)
{
    return is_valid(address.base) && is_valid(address.offset) && is_valid(address.offset_increment);
}

constexpr bool is_valid_immediate_mmio_address(const std::uint32_t byte_address)
{
    return byte_address >= MmioFirstEncodable && byte_address <= MmioLastEncodable && byte_address % 4u == 0u;
}

constexpr bool is_valid_immediate(const MmioAddress address)
{
    return address.addressing == MmioAddressing::Immediate && is_valid_immediate_mmio_address(address.byte_address);
}

constexpr bool is_valid_indirect(const MmioAddress address)
{
    return address.addressing == MmioAddressing::Indirect && is_valid(address.base) && is_valid(address.offset) && is_valid(address.offset_increment);
}

constexpr bool broadcasts_rows(const Broadcast broadcast)
{
    return broadcast == Broadcast::Row || broadcast == Broadcast::Scalar;
}

constexpr bool broadcasts_columns(const Broadcast broadcast)
{
    return broadcast == Broadcast::Column0 || broadcast == Broadcast::Scalar;
}

constexpr std::uint16_t native_rows(const Transfer<SrcA, Dst> transfer)
{
    return transfer.number_of_rows.value % 8u == 0u ? 8u : 1u;
}

constexpr std::uint16_t native_rows(const Transfer<SrcB, Dst> transfer)
{
    const std::uint16_t rows = transfer.number_of_rows.value;

    if (broadcasts_rows(transfer.broadcast) && rows % 8u == 0u)
    {
        return 8u;
    }

    if (!broadcasts_rows(transfer.broadcast) && rows % 4u == 0u)
    {
        return 4u;
    }

    return 1u;
}

constexpr std::uint16_t native_rows(const Transfer<Dst, SrcB> transfer)
{
    return transfer.number_of_rows.value % 4u == 0u ? 4u : 1u;
}

constexpr std::size_t operation_count(const Transfer<SrcA, Dst> transfer)
{
    return transfer.number_of_rows.value / native_rows(transfer);
}

constexpr std::size_t operation_count(const Transfer<SrcB, Dst> transfer)
{
    return transfer.number_of_rows.value / native_rows(transfer);
}

constexpr std::size_t operation_count(const Transfer<Dst, SrcB> transfer)
{
    return transfer.number_of_rows.value / native_rows(transfer);
}

template <typename Source, typename Destination>
constexpr std::size_t operation_count(const Transfer<Source, Destination> transfer)
{
    return transfer.completion == Completion::Wait ? 2u : 1u;
}

constexpr std::size_t operation_count(const Transfer<Gpr, SrcA>)
{
    return 1u;
}

constexpr std::size_t operation_count(const Transfer<Gpr, SrcB>)
{
    return 1u;
}

constexpr bool is_valid(const RowAddressing addressing)
{
    return addressing == RowAddressing::Absolute || addressing == RowAddressing::CounterRelative;
}

// A counter-relative field holds a masked signed offset; only its ten-bit fit is checkable, the
// effective span wraps modulo the counter and is the caller's contract.
constexpr bool is_valid_dst_rows(const std::uint32_t row, const std::uint32_t number_of_rows, const RowAddressing addressing)
{
    return row < MatrixDestinationRows && (addressing == RowAddressing::CounterRelative || row + number_of_rows <= MatrixDestinationRows);
}

constexpr bool has_valid_address_modes(const std::uint8_t address_mode, const std::uint8_t final_address_mode, const std::size_t count)
{
    return address_mode < AddressModeCount && (final_address_mode == SameAddressMode || (final_address_mode < AddressModeCount && count > 1u));
}

constexpr bool is_valid(const Transfer<SrcA, Dst> transfer)
{
    const std::size_t count = operation_count(transfer);
    return transfer.source_row < MatrixSourceRows && is_valid(transfer.destination_addressing) &&
           is_valid_dst_rows(transfer.destination_row, transfer.number_of_rows.value, transfer.destination_addressing) &&
           is_canonical(transfer.number_of_rows) && transfer.source_row + transfer.number_of_rows.value <= MatrixSourceRows &&
           count <= MaxMatrixTransferInstructions && has_valid_address_modes(transfer.address_mode, transfer.final_address_mode, count) &&
           is_valid(transfer.source_validity);
}

constexpr bool is_valid(const Transfer<SrcB, Dst> transfer)
{
    const std::size_t count         = operation_count(transfer);
    const std::uint32_t source_span = broadcasts_rows(transfer.broadcast) ? 1u : transfer.number_of_rows.value;
    return transfer.source_row < MatrixSourceRows && is_valid(transfer.destination_addressing) &&
           is_valid_dst_rows(transfer.destination_row, transfer.number_of_rows.value, transfer.destination_addressing) &&
           is_canonical(transfer.number_of_rows) && transfer.source_row + source_span <= MatrixSourceRows && is_valid(transfer.broadcast) &&
           !(transfer.number_of_rows.value == 1u && transfer.broadcast == Broadcast::Row) && count <= MaxMatrixTransferInstructions &&
           has_valid_address_modes(transfer.address_mode, transfer.final_address_mode, count) && is_valid(transfer.source_validity);
}

constexpr bool is_valid(const Transfer<Dst, SrcB> transfer)
{
    const std::size_t count = operation_count(transfer);
    return is_valid(transfer.source_addressing) && is_valid_dst_rows(transfer.source_row, transfer.number_of_rows.value, transfer.source_addressing) &&
           transfer.destination_row < MatrixSourceRows && is_canonical(transfer.number_of_rows) &&
           transfer.destination_row + transfer.number_of_rows.value <= MatrixSourceRows && count <= MaxMatrixTransferInstructions &&
           has_valid_address_modes(transfer.address_mode, transfer.final_address_mode, count);
}

constexpr bool is_valid(const Transfer<L1, Gpr> transfer)
{
    return is_valid(transfer.source) && is_valid(transfer.destination) && is_valid(transfer.size) && is_valid(transfer.completion) &&
           (transfer.size != ScalarSize::Bytes16 || transfer.destination.index % 4u == 0u);
}

constexpr bool is_valid(const Transfer<Gpr, L1> transfer)
{
    return is_valid(transfer.source) && is_valid(transfer.destination) && is_valid(transfer.size) && is_valid(transfer.completion) &&
           (transfer.size != ScalarSize::Bytes16 || transfer.source.index % 4u == 0u);
}

constexpr bool is_valid(const Transfer<Gpr, SrcA> transfer)
{
    return is_valid(transfer.source) && transfer.source.index % 4u == 0u && is_valid(transfer.destination);
}

constexpr bool is_valid(const Transfer<Gpr, SrcB> transfer)
{
    return is_valid(transfer.source) && transfer.source.index % 4u == 0u && is_valid(transfer.destination);
}

constexpr bool is_valid(const Transfer<Gpr, Mmio> transfer)
{
    return is_valid(transfer.source) && is_valid(transfer.destination.addressing) &&
           (is_valid_immediate(transfer.destination) || is_valid_indirect(transfer.destination)) && is_valid(transfer.completion);
}

constexpr bool is_valid(const Transfer<Mmio, Gpr> transfer)
{
    return is_valid_immediate(transfer.source) && is_valid(transfer.destination) && is_valid(transfer.completion);
}

constexpr bool span_fits_l1(const L1Address16B address, const BlockCount16B size)
{
    return address.block < L1BlockCapacity && size.value > 0u && size.value <= L1BlockCapacity - address.block;
}

constexpr bool is_valid(const Transfer<L1, L1> transfer)
{
    if (!is_valid(transfer.completion) || transfer.size.value == 0u || transfer.size.value > 0xffffu || !span_fits_l1(transfer.source, transfer.size) ||
        !span_fits_l1(transfer.destination, transfer.size))
    {
        return false;
    }

    const std::uint32_t source_end      = transfer.source.block + transfer.size.value;
    const std::uint32_t destination_end = transfer.destination.block + transfer.size.value;
    return source_end <= transfer.destination.block || destination_end <= transfer.source.block;
}

constexpr std::uint32_t encode_movb2d_modifier(const Broadcast broadcast, const std::uint16_t rows)
{
    const bool move_four_rows             = rows == 4u;
    const bool broadcast_one_row_to_eight = rows == 8u && broadcasts_rows(broadcast);

    return (static_cast<std::uint32_t>(move_four_rows) << 2u) | (static_cast<std::uint32_t>(broadcast_one_row_to_eight) << 1u) |
           static_cast<std::uint32_t>(broadcasts_columns(broadcast));
}

constexpr std::uint8_t address_mode_for_fragment(
    const std::uint8_t address_mode, const std::uint8_t final_address_mode, const std::size_t fragment, const std::size_t count)
{
    return fragment + 1u == count && final_address_mode != SameAddressMode ? final_address_mode : address_mode;
}

constexpr std::uint32_t encode(const Transfer<SrcA, Dst> transfer, const std::size_t fragment)
{
    const std::uint32_t opcode       = transfer.source_validity == SourceValidity::Wait ? MoveA2DOpcode : MoveDbgA2DOpcode;
    const std::uint32_t modifier     = native_rows(transfer) == 8u ? 2u : 0u;
    const std::uint32_t address_mode = address_mode_for_fragment(transfer.address_mode, transfer.final_address_mode, fragment, operation_count(transfer));
    const std::uint32_t fields = (static_cast<std::uint32_t>(transfer.destination_32bit_low) << 23u) | (transfer.source_row << 17u) | (address_mode << 14u) |
                                 (modifier << 12u) | transfer.destination_row;
    return encode_instruction(opcode, fields);
}

constexpr std::uint32_t encode(const Transfer<SrcB, Dst> transfer, const std::size_t fragment)
{
    const std::uint32_t opcode       = transfer.source_validity == SourceValidity::Wait ? MoveB2DOpcode : MoveDbgB2DOpcode;
    const std::uint32_t modifier     = encode_movb2d_modifier(transfer.broadcast, native_rows(transfer));
    const std::uint32_t address_mode = address_mode_for_fragment(transfer.address_mode, transfer.final_address_mode, fragment, operation_count(transfer));
    const std::uint32_t fields = (static_cast<std::uint32_t>(transfer.destination_32bit_low) << 23u) | (transfer.source_row << 17u) | (address_mode << 14u) |
                                 (modifier << 11u) | transfer.destination_row;
    return encode_instruction(opcode, fields);
}

constexpr std::uint32_t encode(const Transfer<Dst, SrcB> transfer, const std::size_t fragment)
{
    const std::uint32_t modifier     = native_rows(transfer) == 4u ? 2u : 0u;
    const std::uint32_t address_mode = address_mode_for_fragment(transfer.address_mode, transfer.final_address_mode, fragment, operation_count(transfer));
    const std::uint32_t fields = (static_cast<std::uint32_t>(transfer.source_32bit_low) << 23u) | (transfer.destination_row << 17u) | (address_mode << 14u) |
                                 (modifier << 12u) | transfer.source_row;
    return encode_instruction(MoveD2BOpcode, fields);
}

constexpr std::uint32_t value(const ScalarSize size)
{
    return static_cast<std::uint8_t>(size);
}

constexpr std::uint32_t value(const OffsetIncrement increment)
{
    return static_cast<std::uint8_t>(increment);
}

constexpr std::uint32_t half_register_index(const GprHalfRef half)
{
    return 2u * half.gpr.index + static_cast<std::uint32_t>(half.half == GprHalf::High);
}

constexpr std::uint32_t encode_loadind(const ScalarSize size, const IndirectL1Address source, const GprRef destination)
{
    const std::uint32_t fields = (value(size) << 22u) | (half_register_index(source.offset) << 14u) | (value(source.offset_increment) << 12u) |
                                 (destination.index << 6u) | source.base.index;
    return encode_instruction(LoadIndOpcode, fields);
}

constexpr std::uint32_t encode_storeind(
    const std::uint32_t mode, const GprRef source, const GprRef base, const GprHalfRef offset, const OffsetIncrement increment)
{
    const std::uint32_t fields = (mode << 21u) | (half_register_index(offset) << 14u) | (value(increment) << 12u) | (source.index << 6u) | base.index;
    return encode_instruction(StoreIndOpcode, fields);
}

constexpr std::uint32_t encode(const Transfer<L1, Gpr> transfer)
{
    return encode_loadind(transfer.size, transfer.source, transfer.destination);
}

constexpr std::uint32_t encode(const Transfer<Gpr, L1> transfer)
{
    return encode_storeind(
        4u | value(transfer.size), transfer.source, transfer.destination.base, transfer.destination.offset, transfer.destination.offset_increment);
}

constexpr std::uint32_t encode(const Transfer<Gpr, SrcA> transfer)
{
    return encode_storeind(0u, transfer.source, transfer.destination.base, transfer.destination.offset, transfer.destination.offset_increment);
}

constexpr std::uint32_t encode(const Transfer<Gpr, SrcB> transfer)
{
    return encode_storeind(1u, transfer.source, transfer.destination.base, transfer.destination.offset, transfer.destination.offset_increment);
}

constexpr std::uint32_t immediate_mmio_register_address(const MmioAddress address)
{
    return (address.byte_address - MmioBase) >> 2u;
}

constexpr std::uint32_t encode_storereg(const GprRef source, const MmioAddress destination)
{
    return encode_instruction(StoreRegOpcode, (source.index << 18u) | immediate_mmio_register_address(destination));
}

constexpr std::uint32_t encode_storeind_mmio(const GprRef source, const MmioAddress destination)
{
    return encode_storeind(2u, source, destination.base, destination.offset, destination.offset_increment);
}

constexpr std::uint32_t encode(const Transfer<Gpr, Mmio> transfer)
{
    return transfer.destination.addressing == MmioAddressing::Immediate ? encode_storereg(transfer.source, transfer.destination)
                                                                        : encode_storeind_mmio(transfer.source, transfer.destination);
}

constexpr std::uint32_t encode(const Transfer<Mmio, Gpr> transfer)
{
    return encode_instruction(LoadRegOpcode, (transfer.destination.index << 18u) | immediate_mmio_register_address(transfer.source));
}

constexpr std::uint32_t encode_xmov_launch()
{
    return encode_instruction(XmovOpcode, 0u);
}

constexpr void reject_invalid_constant(const bool valid)
{
    if (__builtin_is_constant_evaluated() && !valid)
    {
        __builtin_trap();
    }
}

#ifdef ENABLE_LLK_ASSERT
inline __attribute__((always_inline)) void assert_valid_relative_row(const std::int32_t offset)
{
    LLK_ASSERT(offset >= -512 && offset < 512, "counter-relative row offset must fit ten signed bits");
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<SrcA, Dst> transfer)
{
    LLK_ASSERT(transfer.source_row < MatrixSourceRows, "MOVA2D source row must be in [0, 63]");
    LLK_ASSERT(is_valid(transfer.destination_addressing), "MOVA2D destination addressing must be Absolute or CounterRelative");
    LLK_ASSERT(transfer.destination_row < MatrixDestinationRows, "MOVA2D destination row field must fit ten bits");
    LLK_ASSERT(is_canonical(transfer.number_of_rows), "MOVA2D row count must be 1, 4, 8, or a positive whole number of faces");
    LLK_ASSERT(transfer.source_row + transfer.number_of_rows.value <= MatrixSourceRows, "MOVA2D source span must fit SrcA");
    LLK_ASSERT(
        transfer.destination_addressing == RowAddressing::CounterRelative || transfer.destination_row + transfer.number_of_rows.value <= MatrixDestinationRows,
        "MOVA2D absolute destination span must fit Dst");
    LLK_ASSERT(operation_count(transfer) <= MaxMatrixTransferInstructions, "MOVA2D transfers may expand to at most 8 native TT instructions");
    LLK_ASSERT(transfer.address_mode < AddressModeCount, "Blackhole MOVA2D address mode must be in [0, 7]");
    LLK_ASSERT(
        transfer.final_address_mode == SameAddressMode || (transfer.final_address_mode < AddressModeCount && operation_count(transfer) > 1u),
        "MOVA2D final address mode must be reused or select [0, 7] for a multi-instruction transfer");
    LLK_ASSERT(is_valid(transfer.source_validity), "MOVA2D source validity must be Wait or Ignore");
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<SrcB, Dst> transfer)
{
    const std::uint32_t source_span = broadcasts_rows(transfer.broadcast) ? 1u : transfer.number_of_rows.value;
    LLK_ASSERT(transfer.source_row < MatrixSourceRows, "MOVB2D source row must be in [0, 63]");
    LLK_ASSERT(is_valid(transfer.destination_addressing), "MOVB2D destination addressing must be Absolute or CounterRelative");
    LLK_ASSERT(transfer.destination_row < MatrixDestinationRows, "MOVB2D destination row field must fit ten bits");
    LLK_ASSERT(is_canonical(transfer.number_of_rows), "MOVB2D row count must be 1, 4, 8, or a positive whole number of faces");
    LLK_ASSERT(transfer.source_row + source_span <= MatrixSourceRows, "MOVB2D source span must fit SrcB");
    LLK_ASSERT(
        transfer.destination_addressing == RowAddressing::CounterRelative || transfer.destination_row + transfer.number_of_rows.value <= MatrixDestinationRows,
        "MOVB2D absolute destination span must fit Dst");
    LLK_ASSERT(is_valid(transfer.broadcast), "MOVB2D broadcast must be None, Column0, Row, or Scalar");
    LLK_ASSERT(transfer.number_of_rows.value != 1u || transfer.broadcast != Broadcast::Row, "one-row MOVB2D cannot request a no-op row broadcast");
    LLK_ASSERT(operation_count(transfer) <= MaxMatrixTransferInstructions, "MOVB2D transfers may expand to at most 8 native TT instructions");
    LLK_ASSERT(transfer.address_mode < AddressModeCount, "Blackhole MOVB2D address mode must be in [0, 7]");
    LLK_ASSERT(
        transfer.final_address_mode == SameAddressMode || (transfer.final_address_mode < AddressModeCount && operation_count(transfer) > 1u),
        "MOVB2D final address mode must be reused or select [0, 7] for a multi-instruction transfer");
    LLK_ASSERT(is_valid(transfer.source_validity), "MOVB2D source validity must be Wait or Ignore");
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<Dst, SrcB> transfer)
{
    LLK_ASSERT(is_valid(transfer.source_addressing), "MOVD2B source addressing must be Absolute or CounterRelative");
    LLK_ASSERT(transfer.source_row < MatrixDestinationRows, "MOVD2B source row field must fit ten bits");
    LLK_ASSERT(transfer.destination_row < MatrixSourceRows, "MOVD2B destination row must be in [0, 63]");
    LLK_ASSERT(is_canonical(transfer.number_of_rows), "MOVD2B row count must be 1, 4, 8, or a positive whole number of faces");
    LLK_ASSERT(
        transfer.source_addressing == RowAddressing::CounterRelative || transfer.source_row + transfer.number_of_rows.value <= MatrixDestinationRows,
        "MOVD2B absolute source span must fit Dst");
    LLK_ASSERT(transfer.destination_row + transfer.number_of_rows.value <= MatrixSourceRows, "MOVD2B destination span must fit SrcB");
    LLK_ASSERT(operation_count(transfer) <= MaxMatrixTransferInstructions, "MOVD2B transfers may expand to at most 8 native TT instructions");
    LLK_ASSERT(transfer.address_mode < AddressModeCount, "Blackhole MOVD2B address mode must be in [0, 7]");
    LLK_ASSERT(
        transfer.final_address_mode == SameAddressMode || (transfer.final_address_mode < AddressModeCount && operation_count(transfer) > 1u),
        "MOVD2B final address mode must be reused or select [0, 7] for a multi-instruction transfer");
}

inline __attribute__((always_inline)) void assert_valid_gpr(const GprRef gpr, [[maybe_unused]] const char* message)
{
    LLK_ASSERT(is_valid(gpr), message);
}

inline __attribute__((always_inline)) void assert_valid_address(const IndirectL1Address address)
{
    assert_valid_gpr(address.base, "indirect L1 base GPR must be in [0, 63]");
    assert_valid_gpr(address.offset.gpr, "indirect L1 offset GPR must be in [0, 63]");
    LLK_ASSERT(is_valid(address.offset.half), "indirect L1 offset half must be Low or High");
    LLK_ASSERT(is_valid(address.offset_increment), "indirect L1 offset increment must be None, Bytes2, Bytes4, or Bytes16");
}

inline __attribute__((always_inline)) void assert_valid_address(const IndirectSrcAddress address)
{
    assert_valid_gpr(address.base, "indirect Src base GPR must be in [0, 63]");
    assert_valid_gpr(address.offset.gpr, "indirect Src offset GPR must be in [0, 63]");
    LLK_ASSERT(is_valid(address.offset.half), "indirect Src offset half must be Low or High");
    LLK_ASSERT(is_valid(address.offset_increment), "indirect Src offset increment must be None, Bytes2, Bytes4, or Bytes16");
}

inline __attribute__((always_inline)) void assert_valid_scalar_common(const ScalarSize size, const Completion completion)
{
    LLK_ASSERT(is_valid(size), "scalar transfer size must be Bytes16, Bytes4, Bytes2, or Bytes1");
    LLK_ASSERT(is_valid(completion), "scalar transfer completion must be IssueOnly or Wait");
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<L1, Gpr> transfer)
{
    assert_valid_address(transfer.source);
    assert_valid_gpr(transfer.destination, "LOADIND destination GPR must be in [0, 63]");
    assert_valid_scalar_common(transfer.size, transfer.completion);
    LLK_ASSERT(transfer.size != ScalarSize::Bytes16 || transfer.destination.index % 4u == 0u, "16-byte LOADIND destination GPR must be four-aligned");
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<Gpr, L1> transfer)
{
    assert_valid_gpr(transfer.source, "STOREIND source GPR must be in [0, 63]");
    assert_valid_address(transfer.destination);
    assert_valid_scalar_common(transfer.size, transfer.completion);
    LLK_ASSERT(transfer.size != ScalarSize::Bytes16 || transfer.source.index % 4u == 0u, "16-byte STOREIND source GPR must be four-aligned");
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<Gpr, SrcA> transfer)
{
    assert_valid_gpr(transfer.source, "STOREIND-to-SrcA source GPR must be in [0, 63]");
    LLK_ASSERT(transfer.source.index % 4u == 0u, "STOREIND-to-SrcA source GPR must be four-aligned");
    assert_valid_address(transfer.destination);
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<Gpr, SrcB> transfer)
{
    assert_valid_gpr(transfer.source, "STOREIND-to-SrcB source GPR must be in [0, 63]");
    LLK_ASSERT(transfer.source.index % 4u == 0u, "STOREIND-to-SrcB source GPR must be four-aligned");
    assert_valid_address(transfer.destination);
}

inline __attribute__((always_inline)) void assert_valid_mmio_address(const MmioAddress address)
{
    LLK_ASSERT(is_valid(address.addressing), "MMIO addressing must be Immediate or Indirect");
    if (address.addressing == MmioAddressing::Immediate)
    {
        LLK_ASSERT(is_valid_immediate_mmio_address(address.byte_address), "immediate MMIO address must be aligned and in [0xffb11000, 0xffbffffc]");
    }
    else
    {
        assert_valid_gpr(address.base, "indirect MMIO base GPR must be in [0, 63]");
        assert_valid_gpr(address.offset.gpr, "indirect MMIO offset GPR must be in [0, 63]");
        LLK_ASSERT(is_valid(address.offset.half), "indirect MMIO offset half must be Low or High");
        LLK_ASSERT(is_valid(address.offset_increment), "indirect MMIO offset increment must be None, Bytes2, Bytes4, or Bytes16");
    }
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<Gpr, Mmio> transfer)
{
    assert_valid_gpr(transfer.source, "MMIO-write source GPR must be in [0, 63]");
    assert_valid_mmio_address(transfer.destination);
    LLK_ASSERT(is_valid(transfer.completion), "MMIO-write completion must be IssueOnly or Wait");
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<Mmio, Gpr> transfer)
{
    LLK_ASSERT(transfer.source.addressing == MmioAddressing::Immediate, "LOADREG supports immediate MMIO addressing only");
    LLK_ASSERT(is_valid_immediate_mmio_address(transfer.source.byte_address), "LOADREG address must be aligned and in [0xffb11000, 0xffbffffc]");
    assert_valid_gpr(transfer.destination, "LOADREG destination GPR must be in [0, 63]");
    LLK_ASSERT(is_valid(transfer.completion), "LOADREG completion must be IssueOnly or Wait");
}

inline __attribute__((always_inline)) void assert_valid(const Transfer<L1, L1> transfer)
{
    LLK_ASSERT(is_valid(transfer.completion), "XMOV completion must be IssueOnly or Wait");
    LLK_ASSERT(transfer.size.value >= 1u && transfer.size.value <= 0xffffu, "XMOV size must be in [1, 65535] 16-byte blocks");
    LLK_ASSERT(span_fits_l1(transfer.source, transfer.size), "XMOV source span must fit Blackhole L1");
    LLK_ASSERT(span_fits_l1(transfer.destination, transfer.size), "XMOV destination span must fit Blackhole L1");
    LLK_ASSERT(
        transfer.source.block + transfer.size.value <= transfer.destination.block || transfer.destination.block + transfer.size.value <= transfer.source.block,
        "overlapping L1-to-L1 XMOV spans are not supported");
}

template <typename Operation>
inline __attribute__((always_inline)) void assert_single_operation(const Operation transfer, [[maybe_unused]] const char* message)
{
    LLK_ASSERT(operation_count(transfer) == 1u, message);
}
#endif

} // namespace detail

/** @brief Return whether a SrcA-to-Dst descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<SrcA, Dst> transfer)
{
    return detail::is_valid(transfer);
}

/** @brief Return whether a SrcB-to-Dst descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<SrcB, Dst> transfer)
{
    return detail::is_valid(transfer);
}

/** @brief Return whether a Dst-to-SrcB descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<Dst, SrcB> transfer)
{
    return detail::is_valid(transfer);
}

/** @brief Return whether an L1-to-GPR descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<L1, Gpr> transfer)
{
    return detail::is_valid(transfer);
}

/** @brief Return whether a GPR-to-L1 descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<Gpr, L1> transfer)
{
    return detail::is_valid(transfer);
}

/** @brief Return whether a GPR-to-SrcA descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<Gpr, SrcA> transfer)
{
    return detail::is_valid(transfer);
}

/** @brief Return whether a GPR-to-SrcB descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<Gpr, SrcB> transfer)
{
    return detail::is_valid(transfer);
}

/** @brief Return whether a GPR-to-MMIO descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<Gpr, Mmio> transfer)
{
    return detail::is_valid(transfer);
}

/** @brief Return whether an MMIO-to-GPR descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<Mmio, Gpr> transfer)
{
    return detail::is_valid(transfer);
}

/** @brief Return whether an L1-to-L1 XMOV descriptor is semantically encodable. */
constexpr bool is_valid(const Transfer<L1, L1> transfer)
{
    return detail::is_valid(transfer);
}

/**
 * @brief Encode a compile-time signed row offset for a counter-relative Dst row field.
 *
 * @tparam Offset: Signed offset added to the row counter, values = <-512..511>.
 */
// Compile-time path
template <std::int32_t Offset>
constexpr std::uint32_t relative_row()
{
    static_assert(Offset >= -512 && Offset < 512, "counter-relative row offset must fit ten signed bits");
    return static_cast<std::uint32_t>(Offset) & 0x3ffu;
}

/**
 * @brief Encode a runtime signed row offset for a counter-relative Dst row field.
 *
 * @param offset: Signed offset added to the row counter, values = <-512..511>.
 */
// Runtime path
inline constexpr __attribute__((always_inline)) std::uint32_t relative_row(const std::int32_t offset)
{
    detail::reject_invalid_constant(offset >= -512 && offset < 512);
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid_relative_row(offset);
    }
#endif
    return static_cast<std::uint32_t>(offset) & 0x3ffu;
}

inline constexpr __attribute__((always_inline)) std::uint32_t Transfer<SrcA, Dst>::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this) && detail::operation_count(*this) == 1u);
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
        detail::assert_single_operation(*this, "multi-instruction MOVA2D descriptors must use run()");
    }
#endif
    return detail::encode(*this, 0u);
}

inline constexpr __attribute__((always_inline)) std::uint32_t Transfer<SrcB, Dst>::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this) && detail::operation_count(*this) == 1u);
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
        detail::assert_single_operation(*this, "multi-instruction MOVB2D descriptors must use run()");
    }
#endif
    return detail::encode(*this, 0u);
}

inline constexpr __attribute__((always_inline)) std::uint32_t Transfer<Dst, SrcB>::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this) && detail::operation_count(*this) == 1u);
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
        detail::assert_single_operation(*this, "multi-instruction MOVD2B descriptors must use run()");
    }
#endif
    return detail::encode(*this, 0u);
}

inline constexpr __attribute__((always_inline)) std::uint32_t Transfer<L1, Gpr>::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this) && completion == Completion::IssueOnly);
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
        detail::assert_single_operation(*this, "completed LOADIND descriptors must use run()");
    }
#endif
    return detail::encode(*this);
}

inline constexpr __attribute__((always_inline)) std::uint32_t Transfer<Gpr, L1>::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this) && completion == Completion::IssueOnly);
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
        detail::assert_single_operation(*this, "completed STOREIND descriptors must use run()");
    }
#endif
    return detail::encode(*this);
}

inline constexpr __attribute__((always_inline)) std::uint32_t Transfer<Gpr, SrcA>::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this));
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
    }
#endif
    return detail::encode(*this);
}

inline constexpr __attribute__((always_inline)) std::uint32_t Transfer<Gpr, SrcB>::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this));
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
    }
#endif
    return detail::encode(*this);
}

inline constexpr __attribute__((always_inline)) std::uint32_t Transfer<Gpr, Mmio>::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this) && completion == Completion::IssueOnly);
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
        detail::assert_single_operation(*this, "completed MMIO-write descriptors must use run()");
    }
#endif
    return detail::encode(*this);
}

inline constexpr __attribute__((always_inline)) std::uint32_t Transfer<Mmio, Gpr>::get_operation() const
{
    detail::reject_invalid_constant(detail::is_valid(*this) && completion == Completion::IssueOnly);
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(*this);
        detail::assert_single_operation(*this, "completed LOADREG descriptors must use run()");
    }
#endif
    return detail::encode(*this);
}

namespace detail
{

template <auto Operation, std::size_t Fragment = 0u>
inline __attribute__((always_inline)) void emit_matrix_transfer()
{
    constexpr std::uint32_t instruction = encode(Operation, Fragment);
    INSTRUCTION_WORD(instruction);

    if constexpr (Fragment + 1u < operation_count(Operation))
    {
        emit_matrix_transfer<Operation, Fragment + 1u>();
    }
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<SrcA, Dst>)
{
    static_assert(detail::is_valid(Operation), "invalid SrcA-to-Dst transfer descriptor");
    if constexpr (detail::is_valid(Operation))
    {
        emit_matrix_transfer<Operation>();
    }
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<SrcB, Dst>)
{
    static_assert(detail::is_valid(Operation), "invalid SrcB-to-Dst transfer descriptor");
    if constexpr (detail::is_valid(Operation))
    {
        emit_matrix_transfer<Operation>();
    }
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<Dst, SrcB>)
{
    static_assert(detail::is_valid(Operation), "invalid Dst-to-SrcB transfer descriptor");
    if constexpr (detail::is_valid(Operation))
    {
        emit_matrix_transfer<Operation>();
    }
}

template <auto Operation>
inline __attribute__((always_inline)) void emit_scalar_transfer()
{
    constexpr std::uint32_t instruction = encode(Operation);
    INSTRUCTION_WORD(instruction);

    if constexpr (Operation.completion == Completion::Wait)
    {
        hal::sync::wait::stall<hal::sync::StallTarget::All, hal::sync::StallCondition::ScalarIdle>();
    }
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<L1, Gpr>)
{
    static_assert(detail::is_valid(Operation), "invalid L1-to-GPR transfer descriptor");
    emit_scalar_transfer<Operation>();
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<Gpr, L1>)
{
    static_assert(detail::is_valid(Operation), "invalid GPR-to-L1 transfer descriptor");
    emit_scalar_transfer<Operation>();
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<Gpr, SrcA>)
{
    static_assert(detail::is_valid(Operation), "invalid GPR-to-SrcA transfer descriptor");
    constexpr std::uint32_t instruction = encode(Operation);
    INSTRUCTION_WORD(instruction);
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<Gpr, SrcB>)
{
    static_assert(detail::is_valid(Operation), "invalid GPR-to-SrcB transfer descriptor");
    constexpr std::uint32_t instruction = encode(Operation);
    INSTRUCTION_WORD(instruction);
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<Gpr, Mmio>)
{
    static_assert(detail::is_valid(Operation), "invalid GPR-to-MMIO transfer descriptor");
    emit_scalar_transfer<Operation>();
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<Mmio, Gpr>)
{
    static_assert(detail::is_valid(Operation), "invalid MMIO-to-GPR transfer descriptor");
    emit_scalar_transfer<Operation>();
}

template <auto Operation>
inline __attribute__((always_inline)) void run_static(const Transfer<L1, L1>)
{
    static_assert(detail::is_valid(Operation), "invalid L1-to-L1 XMOV transfer descriptor");

    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg6].Source_address, hal::cfg::Sec::S0, Operation.source.block>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg6].Destination_address, hal::cfg::Sec::S0, Operation.destination.block>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg6].Buffer_size, hal::cfg::Sec::S0, Operation.size.value>(),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg6].Transfer_direction, hal::cfg::Sec::S0, 3u>());

    constexpr std::uint32_t instruction = encode_xmov_launch();
    INSTRUCTION_WORD(instruction);

    if constexpr (Operation.completion == Completion::Wait)
    {
        hal::sync::wait::stall<hal::sync::StallTarget::All, hal::sync::StallCondition::MoverIdle>();
    }
}

template <std::size_t Fragment = 0u>
inline __attribute__((always_inline)) void emit_runtime_matrix_transfer(
    const std::size_t count, const std::uint32_t intermediate_instruction, const std::uint32_t final_instruction)
{
    if (Fragment < count)
    {
        ckernel::instrn_buffer[0] = Fragment + 1u == count ? final_instruction : intermediate_instruction;

        if constexpr (Fragment + 1u < MaxMatrixTransferInstructions)
        {
            emit_runtime_matrix_transfer<Fragment + 1u>(count, intermediate_instruction, final_instruction);
        }
    }
}

template <typename Source, typename Destination>
inline __attribute__((always_inline)) void run_runtime_matrix(const Transfer<Source, Destination> transfer)
{
    const std::size_t count                      = operation_count(transfer);
    const std::uint32_t intermediate_instruction = encode(transfer, 0u);
    const std::uint32_t final_instruction        = encode(transfer, count - 1u);
    emit_runtime_matrix_transfer(count, intermediate_instruction, final_instruction);
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<SrcA, Dst> transfer)
{
#ifdef ENABLE_LLK_ASSERT
    assert_valid(transfer);
#endif
    run_runtime_matrix(transfer);
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<SrcB, Dst> transfer)
{
#ifdef ENABLE_LLK_ASSERT
    assert_valid(transfer);
#endif
    run_runtime_matrix(transfer);
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<Dst, SrcB> transfer)
{
#ifdef ENABLE_LLK_ASSERT
    assert_valid(transfer);
#endif
    run_runtime_matrix(transfer);
}

template <typename Operation>
inline __attribute__((always_inline)) void run_runtime_scalar(const Operation transfer)
{
#ifdef ENABLE_LLK_ASSERT
    assert_valid(transfer);
#endif
    ckernel::instrn_buffer[0] = encode(transfer);
    if (transfer.completion == Completion::Wait)
    {
        hal::sync::wait::stall(hal::sync::StallTarget::All, hal::sync::StallCondition::ScalarIdle);
    }
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<L1, Gpr> transfer)
{
    run_runtime_scalar(transfer);
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<Gpr, L1> transfer)
{
    run_runtime_scalar(transfer);
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<Gpr, SrcA> transfer)
{
#ifdef ENABLE_LLK_ASSERT
    assert_valid(transfer);
#endif
    ckernel::instrn_buffer[0] = encode(transfer);
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<Gpr, SrcB> transfer)
{
#ifdef ENABLE_LLK_ASSERT
    assert_valid(transfer);
#endif
    ckernel::instrn_buffer[0] = encode(transfer);
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<Gpr, Mmio> transfer)
{
    run_runtime_scalar(transfer);
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<Mmio, Gpr> transfer)
{
    run_runtime_scalar(transfer);
}

inline __attribute__((always_inline)) void run_runtime(const Transfer<L1, L1> transfer)
{
#ifdef ENABLE_LLK_ASSERT
    assert_valid(transfer);
#endif
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg6].Source_address, hal::cfg::Sec::S0>(transfer.source.block),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg6].Destination_address, hal::cfg::Sec::S0>(transfer.destination.block),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg6].Buffer_size, hal::cfg::Sec::S0>(transfer.size.value),
        hal::cfg::set<hal::cfg::Thcon[hal::cfg::Reg6].Transfer_direction, hal::cfg::Sec::S0, 3u>());
    ckernel::instrn_buffer[0] = encode_xmov_launch();

    if (transfer.completion == Completion::Wait)
    {
        hal::sync::wait::stall(hal::sync::StallTarget::All, hal::sync::StallCondition::MoverIdle);
    }
}

} // namespace detail

/**
 * @brief Issue every instruction required by a compile-time transfer descriptor.
 *
 * @tparam Operation Complete constexpr transfer descriptor.
 * @note Configure effective RWC progression and alignment before issuing matrix transfers.
 */
template <auto Operation>
inline __attribute__((always_inline)) void run()
{
    detail::run_static<Operation>(Operation);
}

/**
 * @brief Encode and issue a runtime-selected transfer descriptor.
 *
 * @param operation Complete transfer descriptor.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no validation work.
 */
template <typename Source, typename Destination>
inline __attribute__((always_inline)) void run(const Transfer<Source, Destination> operation)
{
    detail::run_runtime(operation);
}

} // namespace hal::move
