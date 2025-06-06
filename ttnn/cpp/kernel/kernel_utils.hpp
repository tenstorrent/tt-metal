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
template <typename T, uint32_t... I>
T make_runtime_struct_from_args(std::integer_sequence<uint32_t, I...>) {
    static_assert(std::is_aggregate_v<T>, "T must be aggregate-initializable");

    return T{get_arg_val<uint32_t>(I)...};
}

template <typename T>
T make_runtime_struct_from_args() {
    constexpr uint32_t num_fields = sizeof(T) / sizeof(uint32_t);
    static_assert(sizeof(T) % sizeof(uint32_t) == 0, "Struct must be a multiple of uint32_t");
    return make_runtime_struct_from_args<T>(std::make_integer_sequence<uint32_t, num_fields>{});
}

template <typename T, uint32_t... I>
constexpr T make_compile_time_struct_from_args(std::integer_sequence<uint32_t, I...>) {
    static_assert(std::is_aggregate_v<T>, "T must be aggregate-initializable");

    return T{get_compile_time_arg_val(I)...};
}

template <typename T>
constexpr T make_compile_time_struct_from_args() {
    constexpr uint32_t num_fields = sizeof(T) / sizeof(uint32_t);
    static_assert(sizeof(T) % sizeof(uint32_t) == 0, "Struct must be a multiple of uint32_t");
    return make_compile_time_struct_from_args<T>(std::make_integer_sequence<uint32_t, num_fields>{});
}
}  // namespace ttnn::kernel_utils
#endif
