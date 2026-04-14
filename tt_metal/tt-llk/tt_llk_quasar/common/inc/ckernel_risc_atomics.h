// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

inline std::int32_t amomin(std::int32_t *ptr, std::int32_t const against)
{
    std::int32_t old;
    asm volatile("amomin.w %[old], %[against], (%[ptr])\n" : [old] "=r"(old), "+m"(*ptr) : [against] "r"(against), [ptr] "r"(ptr));

    return old;
}

inline std::uint32_t amominu(std::uint32_t *ptr, std::uint32_t const against)
{
    std::uint32_t old;
    asm volatile("amominu.w %[old], %[against], (%[ptr])\n" : [old] "=r"(old), "+m"(*ptr) : [against] "r"(against), [ptr] "r"(ptr));

    return old;
}

inline std::int32_t amomax(std::int32_t *ptr, std::int32_t const against)
{
    std::int32_t old;
    asm volatile("amomax.w %[old], %[against], (%[ptr])\n" : [old] "=r"(old), "+m"(*ptr) : [against] "r"(against), [ptr] "r"(ptr));

    return old;
}

inline std::uint32_t amomaxu(std::uint32_t *ptr, std::uint32_t const against)
{
    std::uint32_t old;
    asm volatile("amomaxu.w %[old], %[against], (%[ptr])\n" : [old] "=r"(old), "+m"(*ptr) : [against] "r"(against), [ptr] "r"(ptr));

    return old;
}

template <typename T, std::enable_if_t<std::is_trivially_copyable_v<T>, int> = 0>
inline T load_acquire(T *ptr)
{
    static_assert(sizeof(T) == sizeof(std::uint32_t), "load_acquire: operand must be 32bit");

    std::uint32_t ret;
    asm volatile("amoadd.w.aq %[ret], zero, (%[ptr])" : [ret] "=r"(ret), "+m"(*ptr) : [ptr] "r"(ptr));

    T result;
    std::memcpy(&result, &ret, sizeof(result));
    return result;
}

template <typename T, std::enable_if_t<std::is_trivially_copyable_v<T>, int> = 0>
inline void store_release(T *ptr, T val)
{
    static_assert(sizeof(T) == sizeof(std::uint32_t), "store_release: operand must be 32bit");

    std::uint32_t val_raw;
    std::memcpy(&val_raw, &val, sizeof(val_raw));

    asm volatile("amoswap.w.rl x0, %[val], (%[ptr])" : "+m"(*ptr) : [val] "r"(val_raw), [ptr] "r"(ptr));
}
