// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// offsets source: tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h
#pragma once

#include <cstdint>

#include "detail/for_each.h"
#include "field.h"
#include "thcon.h"

namespace hal
{
namespace cfg
{

// ============================================================
// UNPACKER
// ============================================================

enum class UnpackerReg : std::uint8_t
{
    Reg0,
    Reg1,
};

enum class BlobContext : std::uint8_t
{
    Context01,
    Context23,
};

class UnpackerAddrCtrlEntry
{
public:
    const Field& Xstride;
    const Field& Ystride;
    const Field& Zstride;
    const Field& Wstride;
};

class UnpackerAddrCtrlFields
{
public:
    std::uint32_t unpacker;

private:
    template <std::uint32_t UnpackerIndex, std::uint32_t RegIndex>
    class Fields
    {
    public:
        static_assert(UnpackerIndex < 2, "unpacker index out of range");
        static_assert(RegIndex < 2, "unpacker register index out of range");

        static constexpr std::uint32_t XyWord = 44 + 12 * RegIndex + 2 * UnpackerIndex;
        static constexpr Field Xstride {
            RegisterFile::State, 32, XyWord, 0, 0, 16, 1, 0}; // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
        static constexpr Field Ystride {
            RegisterFile::State, 32, XyWord, 0, 16, 16, 1, 0}; // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
        static constexpr Field Zstride {
            RegisterFile::State, 32, XyWord + 1, 0, 0, 16, 1, 0}; // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
        static constexpr Field Wstride {
            RegisterFile::State, 32, XyWord + 1, 0, 16, 16, 1, 0}; // Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride (16b)
    };

    [[noreturn]] static UnpackerAddrCtrlEntry invalid_index()
    {
        __builtin_trap();
    }

    template <std::uint32_t UnpackerIndex, std::uint32_t RegIndex>
    static constexpr UnpackerAddrCtrlEntry make()
    {
        return {
            Fields<UnpackerIndex, RegIndex>::Xstride,
            Fields<UnpackerIndex, RegIndex>::Ystride,
            Fields<UnpackerIndex, RegIndex>::Zstride,
            Fields<UnpackerIndex, RegIndex>::Wstride,
        };
    }

    template <std::uint32_t UnpackerIndex>
    static constexpr UnpackerAddrCtrlEntry select(UnpackerReg reg)
    {
        return reg == UnpackerReg::Reg0 ? make<UnpackerIndex, 0>() : reg == UnpackerReg::Reg1 ? make<UnpackerIndex, 1>() : invalid_index();
    }

public:
    template <std::uint32_t RegIndex>
    constexpr UnpackerAddrCtrlEntry operator[](detail::CompileTimeIndex<RegIndex>) const
    {
        static_assert(RegIndex < 2, "unpacker register index out of range");
        return unpacker == 0 ? make<0, RegIndex>() : unpacker == 1 ? make<1, RegIndex>() : invalid_index();
    }

    constexpr UnpackerAddrCtrlEntry operator[](UnpackerReg reg) const
    {
        return unpacker == 0 ? select<0>(reg) : unpacker == 1 ? select<1>(reg) : invalid_index();
    }

    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<2>(static_cast<Function&&>(function));
    }
};

class UnpackerAddrBaseFields
{
public:
    std::uint32_t unpacker;

private:
    template <std::uint32_t UnpackerIndex, std::uint32_t RegIndex>
    class Fields
    {
    public:
        static_assert(UnpackerIndex < 2, "unpacker index out of range");
        static_assert(RegIndex < 2, "unpacker register index out of range");

        static constexpr Field Value {
            RegisterFile::State, 32, 48 + 12 * UnpackerIndex + RegIndex, 0, 0, 18, 1, 0}; // Base 0 (of 0-8) used in X-Y addressing (18b)
    };

    [[noreturn]] static const Field& invalid_index()
    {
        __builtin_trap();
    }

    template <std::uint32_t UnpackerIndex>
    static constexpr const Field& select(UnpackerReg reg)
    {
        return reg == UnpackerReg::Reg0 ? Fields<UnpackerIndex, 0>::Value : reg == UnpackerReg::Reg1 ? Fields<UnpackerIndex, 1>::Value : invalid_index();
    }

public:
    template <std::uint32_t RegIndex>
    constexpr const Field& operator[](detail::CompileTimeIndex<RegIndex>) const
    {
        static_assert(RegIndex < 2, "unpacker register index out of range");
        return unpacker == 0 ? Fields<0, RegIndex>::Value : unpacker == 1 ? Fields<1, RegIndex>::Value : invalid_index();
    }

    constexpr const Field& operator[](UnpackerReg reg) const
    {
        return unpacker == 0 ? select<0>(reg) : unpacker == 1 ? select<1>(reg) : invalid_index();
    }

    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<2>(static_cast<Function&&>(function));
    }
};

class UnpackerBlobsYStartFields
{
public:
    std::uint32_t unpacker;

private:
    static constexpr Field Context01 {RegisterFile::State, 32, 51, 0, 0, 32, 1, 0}; // Context 0&1 blobs y_start (32b)
    static constexpr Field Context23 {RegisterFile::State, 32, 52, 0, 0, 32, 1, 0}; // Context 2&3 blobs y_start (32b)

    [[noreturn]] static const Field& invalid_index()
    {
        __builtin_trap();
    }

public:
    constexpr const Field& operator[](BlobContext context) const
    {
        return unpacker != 0                       ? invalid_index()
               : context == BlobContext::Context01 ? Context01
               : context == BlobContext::Context23 ? Context23
                                                   : invalid_index();
    }
};

class UnpackerContextEntry
{
public:
    const Field& Base;
};

namespace detail
{

template <std::uint32_t UnpackerIndex>
inline constexpr std::uint32_t UnpackerContextCount = UnpackerIndex == 0 ? 8 : 2;

template <std::uint32_t UnpackerIndex, std::uint32_t ContextIndex>
class UnpackerContextDescriptor
{
public:
    static_assert(UnpackerIndex < 2, "unpacker index out of range");
    static_assert(ContextIndex < UnpackerContextCount<UnpackerIndex>, "unpacker context index out of range");

private:
    static constexpr const Field& source()
    {
        if constexpr (ContextIndex == 0)
        {
            return ThconReg3Fields::Base_address;
        }
        else if constexpr (ContextIndex == 1)
        {
            return ThconReg3Fields::Base_cntx1_address;
        }
        else if constexpr (ContextIndex == 2)
        {
            return ThconReg3Fields::Base_cntx2_address;
        }
        else if constexpr (ContextIndex == 3)
        {
            return ThconReg3Fields::Base_cntx3_address;
        }
        else if constexpr (ContextIndex == 4)
        {
            return ThconReg4Fields::Base_cntx4_address;
        }
        else if constexpr (ContextIndex == 5)
        {
            return ThconReg4Fields::Base_cntx5_address;
        }
        else if constexpr (ContextIndex == 6)
        {
            return ThconReg4Fields::Base_cntx6_address;
        }
        else
        {
            return ThconReg4Fields::Base_cntx7_address;
        }
    }

    static constexpr Sec section = UnpackerIndex == 0 ? Sec::S0 : Sec::S1;

public:
    // Resolve the THCON section into a standalone field so Base itself can be
    // used as a single non-type template argument.
    static constexpr Field Base {
        source().file,
        source().wbits,
        source().addr32(section),
        0,
        source().shamt(section),
        source().width,
        1,
        0,
    }; // Unpacker source/tile context base address (aligned to 16B word) (32b)
};

} // namespace detail

class UnpackerContextFields
{
public:
    std::uint32_t unpacker;

private:
    [[noreturn]] static UnpackerContextEntry invalid_index()
    {
        __builtin_trap();
    }

    template <std::uint32_t UnpackerIndex, std::uint32_t ContextIndex = 0>
    static constexpr UnpackerContextEntry select(std::uint32_t context)
    {
        if constexpr (ContextIndex < detail::UnpackerContextCount<UnpackerIndex>)
        {
            return context == ContextIndex ? UnpackerContextEntry {detail::UnpackerContextDescriptor<UnpackerIndex, ContextIndex>::Base}
                                           : select<UnpackerIndex, ContextIndex + 1>(context);
        }
        else
        {
            return invalid_index();
        }
    }

public:
    constexpr UnpackerContextEntry operator[](std::uint32_t context) const
    {
        return unpacker == 0 ? select<0>(context) : unpacker == 1 ? select<1>(context) : invalid_index();
    }
};

template <std::uint32_t UnpackerIndex>
class StaticUnpackerContextFields
{
public:
    static_assert(UnpackerIndex < 2, "unpacker index out of range");

    template <std::uint32_t ContextIndex>
    constexpr UnpackerContextEntry operator[](detail::CompileTimeIndex<ContextIndex>) const
    {
        static_assert(ContextIndex < detail::UnpackerContextCount<UnpackerIndex>, "unpacker context index out of range");
        return {detail::UnpackerContextDescriptor<UnpackerIndex, ContextIndex>::Base};
    }

    constexpr UnpackerContextEntry operator[](std::uint32_t context) const
    {
        return UnpackerContextFields {UnpackerIndex}[context];
    }

    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<detail::UnpackerContextCount<UnpackerIndex>>(static_cast<Function&&>(function));
    }
};

class UnpackerEntry
{
public:
    UnpackerAddrCtrlFields AddrCtrl;
    UnpackerAddrBaseFields AddrBase;
    UnpackerContextFields Cntx;
    const Field& ForcedSharedExp;
    const Field& AddDestAddrCntr;
    const Field& NopRegClrVal;
    UnpackerBlobsYStartFields BlobsYStart;
};

template <std::uint32_t UnpackerIndex>
class StaticUnpackerEntry
{
public:
    UnpackerAddrCtrlFields AddrCtrl;
    UnpackerAddrBaseFields AddrBase;
    StaticUnpackerContextFields<UnpackerIndex> Cntx;
    const Field& ForcedSharedExp;
    const Field& AddDestAddrCntr;
    const Field& NopRegClrVal;
    UnpackerBlobsYStartFields BlobsYStart;
};

class UnpackerFields
{
private:
    template <std::uint32_t UnpackerIndex>
    class Fields
    {
    public:
        static_assert(UnpackerIndex < 2, "unpacker index out of range");

        static constexpr Field ForcedSharedExp {
            RegisterFile::State, 32, 50 + 12 * UnpackerIndex, 0, 0, 8, 1, 0}; // Forced shared exponent used when shared exponent reads are disabled for BFP
                                                                              // formats (8b)
        static constexpr Field AddDestAddrCntr {
            RegisterFile::State, 32, 50 + 12 * UnpackerIndex, 0, 8, 1, 1, 0}; // Combine the per-context destination address with the address counter (1b)
        static constexpr Field NopRegClrVal {RegisterFile::State, 32, 53 + 10 * UnpackerIndex, 0, 0, 32, 1, 0}; // Immediate value used by UNPACR_NOP to clear
                                                                                                                // SRCA (unpacker 0) or SRCB (unpacker 1) (32b)
    };

    [[noreturn]] static UnpackerEntry invalid_index()
    {
        __builtin_trap();
    }

    template <std::uint32_t UnpackerIndex>
    static constexpr UnpackerEntry make()
    {
        return {
            {UnpackerIndex},
            {UnpackerIndex},
            {UnpackerIndex},
            Fields<UnpackerIndex>::ForcedSharedExp,
            Fields<UnpackerIndex>::AddDestAddrCntr,
            Fields<UnpackerIndex>::NopRegClrVal,
            {UnpackerIndex},
        };
    }

    template <std::uint32_t UnpackerIndex>
    static constexpr StaticUnpackerEntry<UnpackerIndex> make_static()
    {
        return {
            {UnpackerIndex},
            {UnpackerIndex},
            {},
            Fields<UnpackerIndex>::ForcedSharedExp,
            Fields<UnpackerIndex>::AddDestAddrCntr,
            Fields<UnpackerIndex>::NopRegClrVal,
            {UnpackerIndex},
        };
    }

public:
    template <std::uint32_t UnpackerIndex>
    constexpr StaticUnpackerEntry<UnpackerIndex> operator[](detail::CompileTimeIndex<UnpackerIndex>) const
    {
        static_assert(UnpackerIndex < 2, "unpacker index out of range");
        return make_static<UnpackerIndex>();
    }

    constexpr UnpackerEntry operator[](std::uint32_t unpacker) const
    {
        return unpacker == 0 ? make<0>() : unpacker == 1 ? make<1>() : invalid_index();
    }

    /**
     * @brief Invoke @p function once for each unpacker at compile time.
     *
     * The callback receives `std::integral_constant<std::uint32_t, I>`, so
     * the index is valid both as an array subscript and in a field expression
     * used as a non-type template argument.
     */
    template <typename Function>
    constexpr void forEach(Function&& function) const
    {
        detail::for_each_index<2>(static_cast<Function&&>(function));
    }
};

/**
 * @brief Compile-time field access for unpacker 0 and unpacker 1 registers.
 *
 * Select the logical unpacker with the first numeric index. Address-control and
 * address-base registers use `UnpackerReg::Reg0` or `UnpackerReg::Reg1`.
 *
 * - `Unpacker[unpacker].AddrCtrl[reg]` exposes `Xstride`, `Ystride`,
 *   `Zstride`, and `Wstride`. Each stride is 16 bits and implements
 *   `Address = Base + X*Xstride + Y*Ystride + Z*Zstride + W*Wstride`.
 * - `Unpacker[unpacker].AddrBase[reg]` is the 18-bit address-space base used
 *   in X-Y addressing.
 * - `Unpacker[unpacker].Cntx[context].Base` is the source/tile base address
 *   for that THCON context. Unpacker 0 has eight contexts; unpacker 1 has two.
 * - `Unpacker[unpacker].ForcedSharedExp` is the 8-bit shared exponent used
 *   when shared-exponent reads are disabled for BFP formats.
 * - `Unpacker[unpacker].AddDestAddrCntr` combines the per-context destination
 *   address with the address counter. It is valid when instruction thread
 *   override mode enables multiple contexts.
 * - `Unpacker[unpacker].NopRegClrVal` is the 32-bit immediate used by
 *   `UNPACR_NOP` to clear SRCA for unpacker 0 or SRCB for unpacker 1.
 * - `Unpacker[0].BlobsYStart[BlobContext::Context01]` and
 *   `Unpacker[0].BlobsYStart[BlobContext::Context23]` select the 32-bit
 *   `blobs_y_start` registers for contexts 0&1 and 2&3. These registers are
 *   used for uncompressed tiles with blobs per XY plane and inline
 *   haloization. Each 4-bit slice sets y_start for one of up to eight blobs.
 *   Unpacker 1 has no blobs-y-start registers.
 *
 * Invalid unpacker indices, enum values, or blob access through unpacker 1
 * fail compilation in a constant-expression/template context and trap at
 * runtime.
 *
 * @code{.cpp}
 * write<Access::TensixCfgUnit,
 *       Unpacker[0].AddrCtrl[UnpackerReg::Reg0].Xstride,
 *       Sec::S0>(x_stride);
 * write<Access::TensixCfgUnit,
 *       Unpacker[1].AddrBase[UnpackerReg::Reg1],
 *       Sec::S0>(base);
 * write<Access::TensixCfgUnit,
 *       Unpacker[0].BlobsYStart[BlobContext::Context01],
 *       Sec::S0>(blobs_y_start);
 *
 * Unpacker.forEach([&](auto U) {
 *     Unpacker[U].Cntx.forEach([&](auto C) {
 *         write<Access::MMIO, Unpacker[U].Cntx[C].Base, Sec::S0>(base[U][C]);
 *     });
 * });
 * @endcode
 */
inline constexpr UnpackerFields Unpacker {};

} // namespace cfg
} // namespace hal
