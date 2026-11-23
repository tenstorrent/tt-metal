// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/tensor_prefetcher_signal.h"
#include "experimental/kernel_args.h"

// Raises one Tensor prefetcher op signal on this device. Runs after every program enqueued before it, so
// their writes have landed by the time the prefetcher sees the signal.
void kernel_main() {
    const uint32_t signal_addr = get_arg(args::signal_addr);
    experimental::tensor_prefetcher_signal(signal_addr);
    noc_async_atomic_barrier();
}
