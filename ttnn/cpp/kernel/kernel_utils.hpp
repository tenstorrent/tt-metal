#pragma once

#if defined(KERNEL_BUILD)

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api/common.h"
#endif

#include <utility>
#include "compile_time_args.h"

namespace ttnn::kernel_utils {
template <typename KernelArgsStruct, uint32_t... I>
KernelArgsStruct make_runtime_struct_from_args(std::integer_sequence<uint32_t, I...>) {
    return KernelArgsStruct{get_arg_val<uint32_t>(I)...};
}

template <typename KernelArgsStruct>
KernelArgsStruct make_runtime_struct_from_args() {
    constexpr uint32_t num_fields = sizeof(KernelArgsStruct) / sizeof(uint32_t);
    return make_runtime_struct_from_args<KernelArgsStruct>(std::make_integer_sequence<uint32_t, num_fields>{});
}

template <typename KernelArgsStruct, uint32_t... I>
constexpr KernelArgsStruct make_compile_time_struct_from_args(std::integer_sequence<uint32_t, I...>) {
    return KernelArgsStruct{get_compile_time_arg_val(I)...};
}

template <typename KernelArgsStruct>
constexpr KernelArgsStruct make_compile_time_struct_from_args() {
    constexpr uint32_t num_fields = sizeof(KernelArgsStruct) / sizeof(uint32_t);
    return make_compile_time_struct_from_args<KernelArgsStruct>(std::make_integer_sequence<uint32_t, num_fields>{});
}
}  // namespace ttnn::kernel_utils
#endif
