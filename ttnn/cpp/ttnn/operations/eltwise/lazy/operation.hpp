// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_stl/small_vector.hpp"

#include <variant>

namespace ttnn::operations::lazy {

// arity is based on number of tensor arguments
// different types should be used to distinguish number of scalar parameters
// e.g. WithParam, WithParam2, etc.

enum class Unary {
    RECIP,
    NEGATIVE,
    EXP,
    EQZ,
    GEZ,
    GTZ,
    LEZ,
    LTZ,
    NEZ,
    LOGICAL_NOT,
};

enum class UnaryWithParam {
    ADD,
    SUB,
    RSUB,
    MUL,
    DIV,
    POWER,
};

enum class Binary {
    ADD,
    SUB,
    MUL,
    DIV,
    POWER,
};

enum class Ternary {
    WHERE,
};

using Operation = std::variant<Unary, UnaryWithParam, Binary, Ternary>;

// inline storage based on max number of tensor arguments
// consider updating if enum class for Quarternary is added
template <typename T>
using Arguments = ttsl::SmallVector<T, 3>;

}  // namespace ttnn::operations::lazy
