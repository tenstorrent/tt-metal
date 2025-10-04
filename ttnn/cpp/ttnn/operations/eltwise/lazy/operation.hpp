// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_stl/small_vector.hpp"

#include <variant>

namespace ttnn::operations::lazy {

enum class Unary {
    ADD,
    SUB,
    RSUB,
    MUL,
    DIV,
    NEGATIVE,
    EXP,
    POWER,
    EQZ,
    GEZ,
    GTZ,
    LEZ,
    LTZ,
    NEZ,
    LOGICAL_NOT,
};

enum class Binary {
    ADD,
    SUB,
    MUL,
};

enum class Ternary {
    WHERE,
};

using Operation = std::variant<Unary, Binary, Ternary>;

template <typename T>
using Arguments = ttsl::SmallVector<T, std::variant_size_v<Operation>>;

}  // namespace ttnn::operations::lazy
