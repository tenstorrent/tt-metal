// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>  // size_t
#include <string>
#include <type_traits>  // is_same_v, decay
#include <utility>      // index_sequence, forward

#include <reflect>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
