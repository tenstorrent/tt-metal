// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/llvm/llvm_small_vector.hpp>

namespace ttsl {

static constexpr size_t SMALL_VECTOR_SIZE = 8;

template <typename T, std::size_t PREALLOCATED_SIZE = SMALL_VECTOR_SIZE>
struct SmallVector : public ttsl::detail::llvm::SmallVector<T, PREALLOCATED_SIZE> {
    using ttsl::detail::llvm::SmallVector<T, PREALLOCATED_SIZE>::SmallVector;
};

}  // namespace ttsl

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt

namespace ttnn {
template <typename T, size_t PREALLOCATED_SIZE = ttsl::SMALL_VECTOR_SIZE>
using SmallVector [[deprecated("Use ttsl::SmallVector instead")]] = ttsl::SmallVector<T, PREALLOCATED_SIZE>;
}
