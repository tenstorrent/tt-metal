// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::distributed {

enum class SocketType : uint8_t { MPI, FABRIC };

enum class EndpointSocketType : uint8_t { SENDER, RECEIVER, BIDIRECTIONAL };

}  // namespace ttnn::distributed
