// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

void kernel_main() {
}
