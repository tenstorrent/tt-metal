// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Compatibility helpers for working with CircularBuffer and
// DataflowBuffer behind a uniform interface in compute helpers.
//
// CircularBuffer exposes get_cb_id() (uint32_t); DataflowBuffer exposes
// get_id() (uint16_t). The buf_id() overload set lets generic helper code
// grab the integer ID for legacy LLK calls regardless of which buffer type
// the caller constructed. Sync methods (reserve_back / push_back / wait_front
// / pop_front) already share names across both classes, so no wrapper needed
// for those.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

namespace compute_kernel_lib {

ALWI uint32_t buf_id(const ::CircularBuffer& cb) { return cb.get_cb_id(); }

ALWI uint32_t buf_id(const ::DataflowBuffer& dfb) { return dfb.get_id(); }

}  // namespace compute_kernel_lib
