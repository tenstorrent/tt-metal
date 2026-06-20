// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Uniform integer-ID access for CircularBuffer and DataflowBuffer in generic compute
// helpers. CircularBuffer exposes get_cb_id(); DataflowBuffer exposes get_id(); the
// buf_id() overloads return the ID for legacy LLK calls regardless of buffer type. The
// sync methods (reserve_back / push_back / wait_front / pop_front) already share names
// across both, so they need no wrapper.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

namespace compute_kernel_lib {

ALWI uint32_t buf_id(const ::CircularBuffer& cb) { return cb.get_cb_id(); }

ALWI uint32_t buf_id(const ::DataflowBuffer& dfb) { return dfb.get_id(); }

}  // namespace compute_kernel_lib
