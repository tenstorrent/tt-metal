// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// SFPI spelling of the vector instructions the generated BF16 kernels schedule by hand.
// Each call names the instruction and its operands as the raw form does, on the named
// LRegs (readlreg/writelreg). The compiler still owns ordering, copies and hazard NOPs,
// so the export accepts a kernel only when its issued stream matches the raw one.
// Replays use SFPI's lltt; predication is a pushc/popc region (Region).
#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu::bf16_sfpi
{
using vec = __xtt_vector;

// Values live in the named LRegs; reads and writes are the registers themselves.
struct Pinned
{
    [[gnu::always_inline]] vec rd(std::uint32_t r) const
    {
        return __builtin_rvtt_sfpreadlreg(r);
    }

    [[gnu::always_inline]] void wr(std::uint32_t r, vec x) const
    {
        __builtin_rvtt_sfpwritelreg(x, r);
    }

    // Only LReg 0..7 take a result; the hardware discards writes to the constants above them.
    [[gnu::always_inline]] void wr_if_writable(std::uint32_t r, vec x) const
    {
        if (r < 8)
        {
            wr(r, x);
        }
    }
};

// A predicated stretch. SFPI lowers a register write as a whole-register copy, so inside
// the region LReg 0..7 are values the compiler predicates; they return to their
// registers once the region closes.
struct Region
{
    vec v[8];

    [[gnu::always_inline]] Region()
    {
        for (std::uint32_t r = 0; r < 8; ++r)
        {
            v[r] = __builtin_rvtt_sfpreadlreg(r);
        }
        __builtin_rvtt_sfppushc(sfpi::SFPPUSHC_MOD1_PUSH);
    }

    [[gnu::always_inline]] void close()
    {
        __builtin_rvtt_sfppopc(sfpi::SFPPOPC_MOD1_POP);
        for (std::uint32_t r = 0; r < 8; ++r)
        {
            __builtin_rvtt_sfpwritelreg(v[r], r);
        }
    }

    [[gnu::always_inline]] vec rd(std::uint32_t r) const
    {
        return r < 8 ? v[r] : __builtin_rvtt_sfpreadlreg(r);
    }

    // An assignment merges into the prior value: disabled lanes keep it (sfpi.h, "Liveness").
    [[gnu::always_inline]] void wr(std::uint32_t r, vec x)
    {
        v[r] = __builtin_rvtt_sfpassign_lv(v[r], x);
    }

    [[gnu::always_inline]] void wr_if_writable(std::uint32_t r, vec x)
    {
        if (r < 8)
        {
            wr(r, x);
        }
    }
};

[[gnu::always_inline]] inline void sfpnop()
{
    __builtin_rvtt_sfpnop();
}

template <class R>
[[gnu::always_inline]] inline void sfpload(std::uint32_t l, std::uint32_t mod0, std::uint32_t mode, std::uint32_t addr, R& f)
{
    f.wr(l, __builtin_rvtt_sfpload(addr, mod0, mode));
}

[[gnu::always_inline]] inline void sfpload(std::uint32_t l, std::uint32_t mod0, std::uint32_t mode, std::uint32_t addr)
{
    Pinned f;
    sfpload(l, mod0, mode, addr, f);
}

// The half-word modes keep the other half, so every load merges into the register.
template <class R>
[[gnu::always_inline]] inline void sfploadi(std::uint32_t l, std::uint32_t mod0, std::uint32_t imm, R& f)
{
    f.wr(l, __builtin_rvtt_sfploadi_lv(ckernel::instrn_buffer, f.rd(l), imm, 0, 0, mod0));
}

[[gnu::always_inline]] inline void sfploadi(std::uint32_t l, std::uint32_t mod0, std::uint32_t imm)
{
    Pinned f;
    sfploadi(l, mod0, imm, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpmad(std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f)
{
    f.wr(d, __builtin_rvtt_sfpmad(f.rd(a), f.rd(b), f.rd(c), m));
}

[[gnu::always_inline]] inline void sfpmad(std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d, std::uint32_t m)
{
    Pinned f;
    sfpmad(a, b, c, d, m, f);
}

// Modifier 9 (VD takes the max, VC the min) has no SFPI value; it is modifier 1 with the operands exchanged.
template <class R>
[[gnu::always_inline]] inline void sfpswap(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f)
{
    if (m == 9)
    {
        auto pair = __builtin_rvtt_sfpswap(f.rd(c), f.rd(d), 1);
        f.wr_if_writable(c, __builtin_rvtt_sfpselect2(pair, 0));
        f.wr_if_writable(d, __builtin_rvtt_sfpselect2(pair, 1));
    }
    else
    {
        auto pair = __builtin_rvtt_sfpswap(f.rd(d), f.rd(c), m);
        f.wr_if_writable(d, __builtin_rvtt_sfpselect2(pair, 0));
        f.wr_if_writable(c, __builtin_rvtt_sfpselect2(pair, 1));
    }
}

[[gnu::always_inline]] inline void sfpswap(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m)
{
    Pinned f;
    sfpswap(a0, c, d, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpsetcc(std::uint32_t, std::uint32_t c, std::uint32_t, std::uint32_t m, R& f)
{
    __builtin_rvtt_sfpsetcc(f.rd(c), m);
}

[[gnu::always_inline]] inline void sfpsetcc(std::uint32_t a0, std::uint32_t c, std::uint32_t a2, std::uint32_t m)
{
    Pinned f;
    sfpsetcc(a0, c, a2, m, f);
}

template <class R>
[[gnu::always_inline]] inline void sfpsetsgn(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f)
{
    if (m & 1)
    {
        f.wr(d, __builtin_rvtt_sfpsetsgn_i(f.rd(c), imm, m & ~1u));
    }
    else
    {
        f.wr(d, __builtin_rvtt_sfpsetsgn_v(f.rd(c), f.rd(d), m));
    }
}

[[gnu::always_inline]] inline void sfpsetsgn(std::uint32_t imm, std::uint32_t c, std::uint32_t d, std::uint32_t m)
{
    Pinned f;
    sfpsetsgn(imm, c, d, m, f);
}

// A lane mask naming all eight lanes (every even bit of the immediate) masks nothing.
template <class R>
[[gnu::always_inline]] inline void sfpconfig(std::uint32_t imm, std::uint32_t dest, std::uint32_t m, R& f)
{
    if ((m & 8) && (imm & 0x5555u) == 0x5555u && !(m & 1))
    {
        m &= ~8u;
    }
    if (m & 1)
    {
        (__builtin_rvtt_sfpwriteconfig_i)(ckernel::instrn_buffer, imm, 0, 0, m & ~1u, dest);
    }
    else
    {
        __builtin_rvtt_sfpwriteconfig_v(f.rd(0), m, dest);
    }
}

[[gnu::always_inline]] inline void sfpconfig(std::uint32_t imm, std::uint32_t dest, std::uint32_t m)
{
    Pinned f;
    sfpconfig(imm, dest, m, f);
}

#if defined(ARCH_BLACKHOLE)
template <class R>
[[gnu::always_inline]] inline void sfparecip(std::uint32_t, std::uint32_t c, std::uint32_t d, std::uint32_t m, R& f)
{
    f.wr(d, __builtin_rvtt_sfparecip(f.rd(c), m));
}

[[gnu::always_inline]] inline void sfparecip(std::uint32_t a0, std::uint32_t c, std::uint32_t d, std::uint32_t m)
{
    Pinned f;
    sfparecip(a0, c, d, m, f);
}

#endif
} // namespace ckernel::sfpu::bf16_sfpi
