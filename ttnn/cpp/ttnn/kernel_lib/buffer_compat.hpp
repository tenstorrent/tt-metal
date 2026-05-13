// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Compatibility helpers for working with experimental::CircularBuffer and
// experimental::DataflowBuffer behind a uniform interface in compute helpers.
//
// CircularBuffer exposes get_cb_id() (uint32_t); DataflowBuffer exposes
// get_id() (uint16_t). The buf_id() overload set lets generic helper code
// grab the integer ID for legacy LLK calls regardless of which buffer type
// the caller constructed. Sync methods (reserve_back / push_back / wait_front
// / pop_front) already share names across both classes, so no wrapper needed
// for those.

#include "experimental/circular_buffer.h"
#include "experimental/dataflow_buffer.h"

namespace compute_kernel_lib {

ALWI uint32_t buf_id(const ::experimental::CircularBuffer& cb) { return cb.get_cb_id(); }

ALWI uint32_t buf_id(const ::experimental::DataflowBuffer& dfb) { return dfb.get_id(); }

}  // namespace compute_kernel_lib
