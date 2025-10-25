// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//===- MemAlloc.h - Memory allocation functions -----------------*- C++ -*-===//
//
// Originally part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file is a modified version of code from llvm/Support/MemAlloc.h.
// Modifications were made by Tenstorrent AI ULC. in 2025 to adapt it for internal use.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines counterparts of C library allocation functions defined in
/// the namespace 'std'. The new allocation functions crash on allocation
/// failure instead of returning null pointer.
///
//===----------------------------------------------------------------------===//
// tt_stl: adapted from MemAlloc.h
//
// Modifications include:
// - Removed LLVM_ATTRIBUTE_RETURNS_NOALIAS
// - Replaced LLVM_ATTRIBUTE_RETURNS_NONNULL with standard C++ equivalents
// - Replaced report_bad_alloc_error with throwing std::bad_alloc
// - Removed unused functions: allocate_buffer, deallocate_buffer
// - Added ttsl::detail:: prefix to LLVM namespace

#pragma once

#include <cstdlib>
#include <stdexcept>

namespace ttsl::detail::llvm {

[[nodiscard]] inline void* safe_malloc(size_t Sz) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    void* Result = std::malloc(Sz);
    if (Result == nullptr) {
        // It is implementation-defined whether allocation occurs if the space
        // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
        // non-zero, if the space requested was zero.
        if (Sz == 0) {
            return safe_malloc(1);
        }
        throw std::bad_alloc();
    }
    return Result;
}

[[nodiscard]] inline void* safe_calloc(size_t Count, size_t Sz) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    void* Result = std::calloc(Count, Sz);
    if (Result == nullptr) {
        // It is implementation-defined whether allocation occurs if the space
        // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
        // non-zero, if the space requested was zero.
        if (Count == 0 || Sz == 0) {
            return safe_malloc(1);
        }
        throw std::bad_alloc();
    }
    return Result;
}

[[nodiscard]] inline void* safe_realloc(void* Ptr, size_t Sz) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    void* Result = std::realloc(Ptr, Sz);
    if (Result == nullptr) {
        // It is implementation-defined whether allocation occurs if the space
        // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
        // non-zero, if the space requested was zero.
        if (Sz == 0) {
            return safe_malloc(1);
        }
        throw std::bad_alloc();
    }
    return Result;
}

}  // namespace ttsl::detail::llvm
