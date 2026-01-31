// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NOLINTBEGIN(misc-include-cleaner)
// Needs shape.hpp to export ttnn::Shape alias to tt_metal::Shape.
#include "ttnn/tensor/shape/shape.hpp"
// Forward include - re-exports tt-metalium storage APIs for TTNN users.
#include <tt-metalium/experimental/tensor/details/storage.hpp>
// NOLINTEND(misc-include-cleaner)
