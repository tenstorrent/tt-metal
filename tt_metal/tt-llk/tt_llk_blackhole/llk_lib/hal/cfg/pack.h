// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// offsets source: tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h
#pragma once

#include <cstdint>

#include "detail/for_each.h"
#include "field.h"

namespace hal
{
namespace cfg
{
// ============================================================
// PACKER
// ============================================================

class Pck0AddrCtrlXyReg0
{ // Packer Address control register 0
public:
    static constexpr Field Xstride {RegisterScope::State, 32, 12, 0, 0, 16, 1, 0};  // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
    static constexpr Field Ystride {RegisterScope::State, 32, 12, 0, 16, 16, 1, 0}; // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
};

class Pck0AddrCtrlZwReg0
{ // Packer Address control register 0
public:
    static constexpr Field Zstride {RegisterScope::State, 32, 13, 0, 0, 16, 1, 0};  // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
    static constexpr Field Wstride {RegisterScope::State, 32, 13, 0, 16, 16, 1, 0}; // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
};

class Pck0AddrCtrlXyReg1
{ // Packer Address control register 1
public:
    static constexpr Field Xstride {RegisterScope::State, 32, 14, 0, 0, 16, 1, 0};  // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
    static constexpr Field Ystride {RegisterScope::State, 32, 14, 0, 16, 16, 1, 0}; // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
};

class Pck0AddrCtrlZwReg1
{ // Packer Address control register 1
public:
    static constexpr Field Zstride {RegisterScope::State, 32, 15, 0, 0, 16, 1, 0};  // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
    static constexpr Field Wstride {RegisterScope::State, 32, 15, 0, 16, 16, 1, 0}; // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
};

class Pck0AddrBaseReg0
{ // Packer address space base register 0
public:
    static constexpr Field Base {RegisterScope::State, 32, 16, 0, 0, 18, 1, 0}; // Base 0 (of 0-8) used in X-Y addressing (18b)
};

class Pck0AddrBaseReg1
{ // Packer address space base register 1
public:
    static constexpr Field Base {RegisterScope::State, 32, 17, 0, 0, 18, 1, 0}; // Base 0 (of 0-8) used in X-Y addressing (18b)
};

enum class PackerReg : std::uint8_t
{
    Reg0,
    Reg1,
};

class PackerAddrCtrlEntry
{
public:
    const Field& Xstride;
    const Field& Ystride;
    const Field& Zstride;
    const Field& Wstride;
};

class PackerAddrCtrlFields
{
private:
    template <std::uint32_t RegIndex>
    static constexpr PackerAddrCtrlEntry make()
    {
        static_assert(RegIndex < 2, "packer address-control register index out of range");
        if constexpr (RegIndex == 0)
        {
            return {
                Pck0AddrCtrlXyReg0::Xstride,
                Pck0AddrCtrlXyReg0::Ystride,
                Pck0AddrCtrlZwReg0::Zstride,
                Pck0AddrCtrlZwReg0::Wstride,
            };
        }
        else
        {
            return {
                Pck0AddrCtrlXyReg1::Xstride,
                Pck0AddrCtrlXyReg1::Ystride,
                Pck0AddrCtrlZwReg1::Zstride,
                Pck0AddrCtrlZwReg1::Wstride,
            };
        }
    }

    [[noreturn]] static PackerAddrCtrlEntry invalid_index()
    {
        __builtin_trap();
    }

public:
    template <std::uint32_t RegIndex>
    constexpr PackerAddrCtrlEntry operator[](detail::CompileTimeIndex<RegIndex>) const
    {
        return make<RegIndex>();
    }

    constexpr PackerAddrCtrlEntry operator[](PackerReg reg) const
    {
        return reg == PackerReg::Reg0 ? make<0>() : reg == PackerReg::Reg1 ? make<1>() : invalid_index();
    }

    constexpr PackerAddrCtrlEntry operator[](std::uint32_t reg) const
    {
        return reg == 0 ? make<0>() : reg == 1 ? make<1>() : invalid_index();
    }

    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<2>(static_cast<Function&&>(function));
    }
};

class PackerAddrBaseFields
{
private:
    template <std::uint32_t RegIndex>
    static constexpr const Field& get()
    {
        static_assert(RegIndex < 2, "packer address-base register index out of range");
        if constexpr (RegIndex == 0)
        {
            return Pck0AddrBaseReg0::Base;
        }
        else
        {
            return Pck0AddrBaseReg1::Base;
        }
    }

    [[noreturn]] static const Field& invalid_index()
    {
        __builtin_trap();
    }

public:
    template <std::uint32_t RegIndex>
    constexpr const Field& operator[](detail::CompileTimeIndex<RegIndex>) const
    {
        return get<RegIndex>();
    }

    constexpr const Field& operator[](PackerReg reg) const
    {
        return reg == PackerReg::Reg0 ? get<0>() : reg == PackerReg::Reg1 ? get<1>() : invalid_index();
    }

    constexpr const Field& operator[](std::uint32_t reg) const
    {
        return reg == 0 ? get<0>() : reg == 1 ? get<1>() : invalid_index();
    }

    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<2>(static_cast<Function&&>(function));
    }
};

class PackerEntry
{
public:
    PackerAddrCtrlFields AddrCtrl;
    PackerAddrBaseFields AddrBase;
};

class PackerFields
{
private:
    [[noreturn]] static PackerEntry invalid_index()
    {
        __builtin_trap();
    }

public:
    template <std::uint32_t PackerIndex>
    constexpr PackerEntry operator[](detail::CompileTimeIndex<PackerIndex>) const
    {
        static_assert(PackerIndex == 0, "Blackhole exposes address configuration only for packer 0");
        return {};
    }

    constexpr PackerEntry operator[](std::uint32_t packer) const
    {
        return packer == 0 ? PackerEntry {} : invalid_index();
    }
};

/**
 * @brief Hierarchical access to packer 0 address-control and address-base fields.
 *
 * Address-control and address-base registers each have two register sets.
 * Select a set with `PackerReg::Reg0`, `PackerReg::Reg1`, a numeric index, or
 * the compile-time index supplied by `forEach()`.
 *
 * @code{.cpp}
 * write<Access::TensixCfgUnit, Packer[0].AddrCtrl[PackerReg::Reg0].Zstride, Sec::S0>(z_stride);
 * write<Access::TensixCfgUnit, Packer[0].AddrBase[PackerReg::Reg1], Sec::S0>(base);
 *
 * Packer[0].AddrCtrl.forEach([&](auto R) {
 *     write<Access::MMIO, Packer[0].AddrCtrl[R].Xstride, Sec::S0>(x_stride[R]);
 * });
 * @endcode
 */
inline constexpr PackerFields Packer {};

class PckDestRdCtrl
{ // Packer dest regs read control
public:
    static constexpr Field Read_32b_data {RegisterScope::State, 32, 18, 0, 0, 1, 1, 0}; // Read 32bit data from dest (fp32 or int32) (1b)
    static constexpr Field Read_unsigned {RegisterScope::State, 32, 18, 0, 1, 1, 1, 0}; // Read unsigned data (applicable with int8 read only) (1b)
    static constexpr Field Read_int8 {RegisterScope::State, 32, 18, 0, 2, 1, 1, 0};     // Read int8 data, produced by SFPU (1b)
    static constexpr Field Round_10b_mant {
        RegisterScope::State, 32, 18, 0, 3, 1, 1, 0}; // Packer gasket rounds to 10b mantissa, regardless of fp_pack format (1b)
};

class PckEdgeTileFaceSetSelect
{ // Packer face set mapping select and enable - 4 reg sets
public:
    static constexpr Field select {RegisterScope::State, 32, 19, 0, 0, 8, 1, 0}; // Select: (8b)
    static constexpr Field enable {RegisterScope::State, 32, 19, 0, 8, 1, 1, 0}; // Enable per face set mapping (1b)
};

class TileRowSetMappingRow
{
public:
    std::uint32_t mapping_index;

private:
    template <std::uint32_t MappingIndex, std::uint32_t SetIndex>
    class Fields
    {
    public:
        static_assert(MappingIndex < 4, "tile row-set mapping index out of range");
        static_assert(SetIndex < 16, "tile row-set set index out of range");

        static constexpr Field Value {RegisterScope::State, 32, 20 + MappingIndex, 0, 2 * SetIndex, 2, 1, 0}; // Two Bit Mask Set Index (2b)
    };

    [[noreturn]] static const Field& invalid_index()
    {
        __builtin_trap();
    }

    template <std::uint32_t MappingIndex, std::uint32_t SetIndex = 0>
    static constexpr const Field& select(std::uint32_t set_index)
    {
        if constexpr (SetIndex < 16)
        {
            return set_index == SetIndex ? Fields<MappingIndex, SetIndex>::Value : select<MappingIndex, SetIndex + 1>(set_index);
        }
        else
        {
            return invalid_index();
        }
    }

public:
    template <std::uint32_t SetIndex>
    constexpr const Field& operator[](detail::CompileTimeIndex<SetIndex>) const
    {
        static_assert(SetIndex < 16, "tile row-set set index out of range");
        return mapping_index == 0   ? Fields<0, SetIndex>::Value
               : mapping_index == 1 ? Fields<1, SetIndex>::Value
               : mapping_index == 2 ? Fields<2, SetIndex>::Value
               : mapping_index == 3 ? Fields<3, SetIndex>::Value
                                    : invalid_index();
    }

    constexpr const Field& operator[](std::uint32_t set_index) const
    {
        return mapping_index == 0   ? select<0>(set_index)
               : mapping_index == 1 ? select<1>(set_index)
               : mapping_index == 2 ? select<2>(set_index)
               : mapping_index == 3 ? select<3>(set_index)
                                    : invalid_index();
    }

    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<16>(static_cast<Function&&>(function));
    }
};

class TileRowSetMappingFields
{
private:
    [[noreturn]] static TileRowSetMappingRow invalid_index()
    {
        __builtin_trap();
    }

public:
    template <std::uint32_t MappingIndex>
    constexpr TileRowSetMappingRow operator[](detail::CompileTimeIndex<MappingIndex>) const
    {
        static_assert(MappingIndex < 4, "tile row-set mapping index out of range");
        return {MappingIndex};
    }

    constexpr TileRowSetMappingRow operator[](std::uint32_t mapping_index) const
    {
        return mapping_index < 4 ? TileRowSetMappingRow {mapping_index} : invalid_index();
    }

    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<4>(static_cast<Function&&>(function));
    }
};

/**
 * @brief Two-dimensional compile-time access to tile row-set mappings.
 *
 * The first index selects one of four mapping registers at Tensix CFG words
 * 20 through 23. The second index selects one of sixteen 2-bit mask-set fields
 * in that word; set index N has shift `2 * N`.
 *
 * Both indices are bounds checked. In a constant-expression/template context,
 * mapping indices outside 0..3 or set indices outside 0..15 fail compilation.
 * Invalid runtime indices trap instead of producing an out-of-bounds access.
 *
 * @code{.cpp}
 * write<Access::TensixCfgUnit, TileRowSetMapping[2][5], Sec::S0>(mask_set_index);
 *
 * TileRowSetMapping.forEach([&](auto M) {
 *     TileRowSetMapping[M].forEach([&](auto S) {
 *         write<Access::MMIO, TileRowSetMapping[M][S], Sec::S0>(values[M][S]);
 *     });
 * });
 * @endcode
 */
inline constexpr TileRowSetMappingFields TileRowSetMapping {};

class PckEdgeOffsetSec0
{ // Packer edge offset masks
public:
    static constexpr Field mask {RegisterScope::State, 32, 24, 0, 0, 16, 1, 0}; // Row mask (16b)
};

class PckEdgeMode
{ // Packer edge offset mode
public:
    static constexpr Field mode {RegisterScope::State, 32, 24, 0, 16, 1, 1, 0}; // Mode: (1b)
};

class PckEdgeTileRowSetSelect
{ // Packer row set mapping select - 4 reg sets
public:
    static constexpr Field select {RegisterScope::State, 32, 24, 0, 17, 8, 1, 0}; // Select: (8b)
};

class PckEdgeOffsetSec1
{ // Packer edge offset masks
public:
    static constexpr Field mask {RegisterScope::State, 32, 25, 0, 0, 16, 1, 0}; // Row mask (16b)
};

class PckEdgeOffsetSec2
{ // Packer edge offset masks
public:
    static constexpr Field mask {RegisterScope::State, 32, 26, 0, 0, 16, 1, 0}; // Row mask (16b)
};

class PckEdgeOffsetSec3
{ // Packer edge offset masks
public:
    static constexpr Field mask {RegisterScope::State, 32, 27, 0, 0, 16, 1, 0}; // Row mask (16b)
};

class PackCounters
{ // These registers are used to control Z-mask calculatios, auto-generated 'last word' bit, and tile position generator, which, in turn, controls bias and edge
public:
    // masking.
    static constexpr Field pack_per_xy_plane {RegisterScope::State, 32, 28, 0, 0, 8, 4, 32}; // Number of pack instructions per one XY plane (8b)
    static constexpr Field pack_reads_per_xy_plane {
        RegisterScope::State, 32, 28, 0, 8, 8, 4, 32}; // Number of pack reads from destination registers per XY plane (8b)
    static constexpr Field pack_xys_per_tile {RegisterScope::State, 32, 28, 0, 16, 7, 4, 32};  // Number of XY planes in one tile (7b)
    static constexpr Field pack_yz_transposed {RegisterScope::State, 32, 28, 0, 23, 1, 4, 32}; // Tile position counts transposed Y/Z (1b)
    static constexpr Field auto_ctxt_inc_xys_cnt {
        RegisterScope::State, 32, 28, 0, 24, 8, 4, 32}; // Number of xy planes after which Packer context flops will  be incremented (8b)
};

class PackConcatMask
{ // Concat mask per xy plane for blob packing Pack per face edge mask select mapping
public:
    static constexpr Field pack_concat_mask {RegisterScope::State, 32, 32, 0, 0, 16, 4, 32}; // Concat mask per xy plane (16b)
};

class TileFaceSetMappingRow
{
public:
    std::uint32_t mapping_index;

private:
    template <std::uint32_t MappingIndex, std::uint32_t SetIndex>
    class Fields
    {
    public:
        static_assert(MappingIndex < 4, "tile face-set mapping index out of range");
        static_assert(SetIndex < 16, "tile face-set set index out of range");

        static constexpr Field Value {RegisterScope::State, 32, 36 + MappingIndex, 0, 2 * SetIndex, 2, 1, 0}; // Two Bit Face Mask Set Index (2b)
    };

    [[noreturn]] static const Field& invalid_index()
    {
        __builtin_trap();
    }

    template <std::uint32_t MappingIndex, std::uint32_t SetIndex = 0>
    static constexpr const Field& select(std::uint32_t set_index)
    {
        if constexpr (SetIndex < 16)
        {
            return set_index == SetIndex ? Fields<MappingIndex, SetIndex>::Value : select<MappingIndex, SetIndex + 1>(set_index);
        }
        else
        {
            return invalid_index();
        }
    }

public:
    template <std::uint32_t SetIndex>
    constexpr const Field& operator[](detail::CompileTimeIndex<SetIndex>) const
    {
        static_assert(SetIndex < 16, "tile face-set set index out of range");
        return mapping_index == 0   ? Fields<0, SetIndex>::Value
               : mapping_index == 1 ? Fields<1, SetIndex>::Value
               : mapping_index == 2 ? Fields<2, SetIndex>::Value
               : mapping_index == 3 ? Fields<3, SetIndex>::Value
                                    : invalid_index();
    }

    constexpr const Field& operator[](std::uint32_t set_index) const
    {
        return mapping_index == 0   ? select<0>(set_index)
               : mapping_index == 1 ? select<1>(set_index)
               : mapping_index == 2 ? select<2>(set_index)
               : mapping_index == 3 ? select<3>(set_index)
                                    : invalid_index();
    }

    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<16>(static_cast<Function&&>(function));
    }
};

class TileFaceSetMappingFields
{
private:
    [[noreturn]] static TileFaceSetMappingRow invalid_index()
    {
        __builtin_trap();
    }

public:
    template <std::uint32_t MappingIndex>
    constexpr TileFaceSetMappingRow operator[](detail::CompileTimeIndex<MappingIndex>) const
    {
        static_assert(MappingIndex < 4, "tile face-set mapping index out of range");
        return {MappingIndex};
    }

    constexpr TileFaceSetMappingRow operator[](std::uint32_t mapping_index) const
    {
        return mapping_index < 4 ? TileFaceSetMappingRow {mapping_index} : invalid_index();
    }

    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<4>(static_cast<Function&&>(function));
    }
};

/**
 * @brief Two-dimensional compile-time access to tile face-set mappings.
 *
 * The first index selects one of four mapping registers at Tensix CFG words
 * 36 through 39. The second index selects one of sixteen 2-bit face-mask-set
 * fields in that word; set index N has shift `2 * N`.
 *
 * Both indices are bounds checked. In a constant-expression/template context,
 * mapping indices outside 0..3 or set indices outside 0..15 fail compilation.
 * Invalid runtime indices trap instead of producing an out-of-bounds access.
 *
 * @code{.cpp}
 * write<Access::TensixCfgUnit, TileFaceSetMapping[2][5], Sec::S0>(face_mask_set_index);
 *
 * TileFaceSetMapping.forEach([&](auto M) {
 *     TileFaceSetMapping[M].forEach([&](auto S) {
 *         write<Access::MMIO, TileFaceSetMapping[M][S], Sec::S0>(values[M][S]);
 *     });
 * });
 * @endcode
 */
inline constexpr TileFaceSetMappingFields TileFaceSetMapping {};

class PackGlobalCfgCtl
{ // Global packer config control across all contexts
public:
    static constexpr Field pack_disable_fast_tile_end_drain {
        RegisterScope::State, 32, 40, 0, 0, 1, 1, 0}; // Disable fast tile end drain for PackerConcat mask per xy plane for blob packing (1b)
};

} // namespace cfg
} // namespace hal
