// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(KERNEL_BUILD)

#include <type_traits>
#include <reflect>
#include <vector>
#include <bit>

namespace ttnn::kernel_utils {

template <typename KernelStruct>
concept KernelArgsStructU32Concept =
    alignof(KernelStruct) == alignof(uint32_t) && sizeof(KernelStruct) % sizeof(uint32_t) == 0 &&
    std::is_aggregate_v<KernelStruct> && [] {
        bool result = true;
        KernelStruct val{};
        reflect::for_each(
            [&](auto I) { result &= reflect::type_name(uint32_t{}) == reflect::type_name(reflect::get<I>(val)); }, val);
        return result;
    }();

template <KernelArgsStructU32Concept KernStruct>
consteval uint32_t amount_of_fields() {
    return sizeof(KernStruct) / sizeof(uint32_t);
}

template <KernelArgsStructU32Concept KernStruct>
std::vector<uint32_t> to_vector(const KernStruct& kernel_data) {
    constexpr auto N = amount_of_fields<KernStruct>();
    auto res = std::bit_cast<std::array<std::uint32_t, N>>(kernel_data);
    return std::vector<uint32_t>(res.begin(), res.end());
}

}  // namespace ttnn::kernel_utils

#define VALIDATE_KERNEL_ARGS_STRUCT(KernStruct)                     \
    static_assert(                                                  \
        ttnn::kernel_utils::KernelArgsStructU32Concept<KernStruct>, \
        "Struct does not satisfy the requirements of KernelArgsStructU32Concept.");

#else
#define VALIDATE_KERNEL_ARGS_STRUCT(KernStruct)
#endif
