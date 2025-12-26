// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <functional>
#include "tt_stl/guard.hpp"

namespace ttnn {

using ttsl::Guard;
using ttsl::make_guard;
using ttsl::ScopeGuard;

}  // namespace ttnn
