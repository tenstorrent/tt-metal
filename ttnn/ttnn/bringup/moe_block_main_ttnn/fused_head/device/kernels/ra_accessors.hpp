// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routing kernels' DRAM tensors' accessor args, after the kernel's own compile-time args from RA_ACC_CT on
// (head_program.cpp: ChipProgram::routing_accessors), in this order. The kernels pass their own page sizes (the plan's
// 4 KB, a slot, a counts row), which may differ from the tensor's.
#pragma once

#include "api/tensor/tensor_accessor.h"

namespace ra_acc {
constexpr auto plan = TensorAccessorArgs<RA_ACC_CT>();
constexpr auto counts = TensorAccessorArgs<plan.next_compile_time_args_offset()>();
constexpr auto regions = TensorAccessorArgs<counts.next_compile_time_args_offset()>();
constexpr auto token_index = TensorAccessorArgs<regions.next_compile_time_args_offset()>();
constexpr auto table = TensorAccessorArgs<token_index.next_compile_time_args_offset()>();
constexpr auto slots = TensorAccessorArgs<table.next_compile_time_args_offset()>();   // the exchange's DRAM slots
constexpr auto fslots = TensorAccessorArgs<slots.next_compile_time_args_offset()>();  // its forwarding slots
constexpr auto start = TensorAccessorArgs<fslots.next_compile_time_args_offset()>();  // a padded chunk's actual_start
constexpr auto end = TensorAccessorArgs<start.next_compile_time_args_offset()>();     // its actual_end
}  // namespace ra_acc
