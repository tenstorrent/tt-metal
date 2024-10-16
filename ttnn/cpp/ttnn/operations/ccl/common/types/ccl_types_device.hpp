// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types.hpp"

namespace ttnn {
namespace ccl {

template <>
auto build_from_args<WorkerEdmInterfaceArgs>(std::size_t &rt_arg_idx) -> WorkerEdmInterfaceArgs {
    static_assert(sizeof(address_t) <= sizeof(uint32_t), "Address type is too large for this function.");
    return WorkerEdmInterfaceArgs{get_arg_val<uint32_t>(rt_arg_idx++),
                                  get_arg_val<uint32_t>(rt_arg_idx++),
                                  reinterpret_cast<address_t>(get_arg_val<uint32_t>(rt_arg_idx++)),
                                  reinterpret_cast<address_t>(get_arg_val<uint32_t>(rt_arg_idx++)),
                                  get_arg_val<uint32_t>(rt_arg_idx++)};
}

template <>
constexpr std::size_t ct_args_consumed<WorkerEdmInterfaceArgs>() {
    return 0;
}

}  // namespace ccl
}  // namespace ttnn
