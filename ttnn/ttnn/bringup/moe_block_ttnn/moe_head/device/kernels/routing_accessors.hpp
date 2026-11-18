// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routing exchange's DRAM tensors' accessor args, for every routing kernel (the planner's - routing_planner.hpp -
// and the exchange's senders - router_fabric_sends.hpp).
#pragma once

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/tensor/tensor_accessor.h"
#include "head_named_args.hpp"

// The routing kernels' DRAM tensors' accessor args, after the kernel's own positional compile-time args, from the named
// arg routing_accessors_compile_time_args_offset on (head_chip_program.hpp: ChipProgram::routing_accessors), in this
// order - the accessor args are positional compile-time args by the TensorAccessorArgs API. The kernels pass their own
// page sizes (the plan's 4 KB, a slot, a counts row), which may differ from the tensor's.
namespace routing_accessors {
constexpr uint32_t offset = get_named_compile_time_arg_val(moe_head_named_args::kRoutingAccessorsOffset);
constexpr auto plan = TensorAccessorArgs<offset>();
constexpr auto counts = TensorAccessorArgs<plan.next_compile_time_args_offset()>();
constexpr auto regions = TensorAccessorArgs<counts.next_compile_time_args_offset()>();
constexpr auto token_index = TensorAccessorArgs<regions.next_compile_time_args_offset()>();
constexpr auto table = TensorAccessorArgs<token_index.next_compile_time_args_offset()>();
constexpr auto slots = TensorAccessorArgs<table.next_compile_time_args_offset()>();  // the exchange's DRAM slots
constexpr auto start = TensorAccessorArgs<slots.next_compile_time_args_offset()>();  // a padded chunk's actual_start
constexpr auto end = TensorAccessorArgs<start.next_compile_time_args_offset()>();    // its actual_end
}  // namespace routing_accessors
