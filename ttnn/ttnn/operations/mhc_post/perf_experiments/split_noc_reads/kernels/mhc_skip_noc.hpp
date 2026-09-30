// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// perf-experiment ablation: with MHC_SKIP_NOC the F / X / X' data transfers are dropped (CB handshakes and
// barriers kept), so the compute kernel runs against a zero-cost DM pipeline. No effect otherwise.
#pragma once

#include "api/dataflow/dataflow_api.h"

template <typename... A>
FORCE_INLINE void data_read(A... a) {
#ifndef MHC_SKIP_NOC
    noc_async_read(a...);
#endif
}
template <typename... A>
FORCE_INLINE void data_write(A... a) {
#ifndef MHC_SKIP_NOC
    noc_async_write(a...);
#endif
}
