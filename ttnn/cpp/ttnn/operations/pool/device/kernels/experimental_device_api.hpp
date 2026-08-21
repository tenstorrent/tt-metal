// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Convenience header that includes all device 2.0 APIs
// and provides short type aliases for kernel code in conv/pool operations.
// Safe to include from both dataflow and compute (TRISC) kernels.

#pragma once

#include "api/dataflow/circular_buffer.h"

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#endif

namespace experimental {

// Short alias for CircularBuffer (available on both dataflow and compute)
using CB = CircularBuffer;

#ifndef COMPILE_FOR_TRISC
// Short aliases for NOC types (dataflow kernels only)
// Note: CoreLocalMem<uint32_t> is used internally by the read_with_state raw-address overload.
// Prefer passing experimental::CB with offset_bytes args over constructing CoreLocalMem directly.

// The local L1 -> L1 copy helpers now live in the kernel library:
//   ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp
// which is the single source of truth for their contracts (why a self-aimed READ and not a write,
// the per-NoC my_x/my_y rule, the single-packet bound, and the deliberate size_bytes = 0 in the
// typed-destination overloads). These using-declarations keep the historical
// `experimental::`-qualified spelling working for existing conv/pool/sdpa/fold call sites; NEW
// call sites should include the kernel_lib header directly and say `dataflow_kernel_lib::`.
using dataflow_kernel_lib::async_read_barrier_with_trid;
using dataflow_kernel_lib::local_addr;
using dataflow_kernel_lib::read_with_state;
using dataflow_kernel_lib::set_read_state;
using dataflow_kernel_lib::set_read_trid;

#endif

}  // namespace experimental
