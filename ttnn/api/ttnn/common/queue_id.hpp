// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt_stl/strong_type.hpp>
#include <cstdint>

namespace ttnn {

using QueueId = ttsl::StrongType<uint8_t, struct QueueIdTag>;

}  // namespace ttnn
