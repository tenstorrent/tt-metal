// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_protocol.hpp"

namespace dataflow_kernel_lib {

// Guard protects source L1 before returning. CallerManaged lets the caller provide that
// protection; modes may still need completion barriers for data-before-ready ordering.
enum class SourceL1Guard { Guard, CallerManaged };

}  // namespace dataflow_kernel_lib
