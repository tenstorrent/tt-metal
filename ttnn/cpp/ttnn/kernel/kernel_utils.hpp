// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(KERNEL_BUILD)

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api/common.h"
#endif

#include <utility>
#include "compile_time_args.h"
#include "ttnn/kernel/kernel_common_utils.hpp"  // SerializableKernelArgs
#include <bit>

namespace ttnn::kernel_utils {
template <typename KernelArgsStruct, uint32_t... I>
KernelArgsStruct make_runtime_struct_from_args(std::integer_sequence<uint32_t, I...>) {
    static_assert(
        ttnn::kernel_utils::SerializableKernelArgs<KernelArgsStruct>,
        "Struct does not satisfy the requirements of SerializableKernelArgs concept.");
    const uint32_t args[]{get_arg_val<uint32_t>(I)...};
#if defined(__cpp_lib_bit_cast) && (__cpp_lib_bit_cast >= 201806L)
    return std::bit_cast<KernelArgsStruct>(args);
#else
    return __builtin_bit_cast(KernelArgsStruct, args);
#endif
}

template <typename KernelArgsStruct>
KernelArgsStruct make_runtime_struct_from_args() {
    static_assert(
        ttnn::kernel_utils::SerializableKernelArgs<KernelArgsStruct>,
        "Struct does not satisfy the requirements of SerializableKernelArgs concept.");
    constexpr uint32_t num_fields = sizeof(KernelArgsStruct) / sizeof(uint32_t);
    return make_runtime_struct_from_args<KernelArgsStruct>(std::make_integer_sequence<uint32_t, num_fields>{});
}

template <typename KernelArgsStruct, uint32_t... I>
constexpr KernelArgsStruct make_compile_time_struct_from_args(std::integer_sequence<uint32_t, I...>) {
    static_assert(
        ttnn::kernel_utils::SerializableKernelArgs<KernelArgsStruct>,
        "Struct does not satisfy the requirements of SerializableKernelArgs concept.");
    constexpr uint32_t args[]{get_compile_time_arg_val(I)...};
#if defined(__cpp_lib_bit_cast) && (__cpp_lib_bit_cast >= 201806L)
    return std::bit_cast<KernelArgsStruct>(args);
#else
    return __builtin_bit_cast(KernelArgsStruct, args);
#endif
}

template <typename KernelArgsStruct>
constexpr KernelArgsStruct make_compile_time_struct_from_args() {
    static_assert(
        ttnn::kernel_utils::SerializableKernelArgs<KernelArgsStruct>,
        "Struct does not satisfy the requirements of SerializableKernelArgs concept.");
    constexpr uint32_t num_fields = sizeof(KernelArgsStruct) / sizeof(uint32_t);
    return make_compile_time_struct_from_args<KernelArgsStruct>(std::make_integer_sequence<uint32_t, num_fields>{});
}
}  // namespace ttnn::kernel_utils
#endif
