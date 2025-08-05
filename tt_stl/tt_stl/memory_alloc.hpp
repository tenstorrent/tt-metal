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

#ifndef LLVM_SUPPORT_MEMALLOC_H
#define LLVM_SUPPORT_MEMALLOC_H

#include <cstdlib>
#include <stdexcept>

namespace ttsl::detail::llvm {

inline void* safe_malloc(size_t Sz) {
    void* Result = std::malloc(Sz);
    if (Result == nullptr) {
        // It is implementation-defined whether allocation occurs if the space
        // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
        // non-zero, if the space requested was zero.
        if (Sz == 0) {
            return safe_malloc(1);
        }
        throw std::runtime_error("Allocation failed");
    }
    return Result;
}

inline void* safe_calloc(size_t Count, size_t Sz) {
    void* Result = std::calloc(Count, Sz);
    if (Result == nullptr) {
        // It is implementation-defined whether allocation occurs if the space
        // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
        // non-zero, if the space requested was zero.
        if (Count == 0 || Sz == 0) {
            return safe_malloc(1);
        }
        throw std::runtime_error("Allocation failed");
    }
    return Result;
}

inline void* safe_realloc(void* Ptr, size_t Sz) {
    void* Result = std::realloc(Ptr, Sz);
    if (Result == nullptr) {
        // It is implementation-defined whether allocation occurs if the space
        // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
        // non-zero, if the space requested was zero.
        if (Sz == 0) {
            return safe_malloc(1);
        }
        throw std::runtime_error("Allocation failed");
    }
    return Result;
}

/// Allocate a buffer of memory with the given size and alignment.
///
/// When the compiler supports aligned operator new, this will use it to
/// handle even over-aligned allocations.
///
/// However, this doesn't make any attempt to leverage the fancier techniques
/// like posix_memalign due to portability. It is mostly intended to allow
/// compatibility with platforms that, after aligned allocation was added, use
/// reduced default alignment.
void* allocate_buffer(size_t Size, size_t Alignment);

/// Deallocate a buffer of memory with the given size and alignment.
///
/// If supported, this will used the sized delete operator. Also if supported,
/// this will pass the alignment to the delete operator.
///
/// The pointer must have been allocated with the corresponding new operator,
/// most likely using the above helper.
void deallocate_buffer(void* Ptr, size_t Size, size_t Alignment);

}  // namespace ttsl::detail::llvm
#endif
