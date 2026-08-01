// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// comm_skeleton probe: THE INTERCEPT. An empty dataflow kernel deployed on the same core set as
// every other probe. Its DEVICE KERNEL DURATION is the fixed per-dispatch launch/teardown cost that
// must be subtracted from every other probe's absolute number before a slope is read as "ns per
// operation". Nothing else in this file — that is the point.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {}
