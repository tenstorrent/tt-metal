// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The op's DRAM tensors' accessor args: every data-movement kernel's positional compile-time args, from index 0
// (flat_routed_expert_program_factory.cpp), in this order - all its other compile-time args are named
// (flat_options.hpp). A TensorAccessorArgs chain is offset-based, so these stay positional. The kernels pass their own
// page sizes (a row, a word table, the counts row), which may differ from the tensor's.
#pragma once

#include "api/tensor/tensor_accessor.h"
#include "flat_options.hpp"

namespace flat_acc {
constexpr auto x = TensorAccessorArgs<0>();                                       // the dispatch buffer
constexpr auto y = TensorAccessorArgs<x.next_compile_time_args_offset()>();       // the output
constexpr auto counts = TensorAccessorArgs<y.next_compile_time_args_offset()>();  // tokens per global expert
constexpr auto regions = TensorAccessorArgs<counts.next_compile_time_args_offset()>();
constexpr auto ids = TensorAccessorArgs<regions.next_compile_time_args_offset()>();  // the local experts' global ids
constexpr auto token_index = TensorAccessorArgs<ids.next_compile_time_args_offset()>();
constexpr auto y_table = TensorAccessorArgs<token_index.next_compile_time_args_offset()>();  // the combine's row tables
}  // namespace flat_acc
