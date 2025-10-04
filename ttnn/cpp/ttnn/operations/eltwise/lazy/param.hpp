// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_stl/small_vector.hpp"

#include <variant>

namespace ttnn::operations::lazy {

using Param = std::variant<float, std::int32_t, std::uint32_t>;
using Params = ttsl::SmallVector<Param, 1>;
using ParamsView = std::span<const Param>;

}  // namespace ttnn::operations::lazy
