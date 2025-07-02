// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ttnn::kernel_utils {
#if !defined(KERNEL_BUILD)

#include <type_traits>
#include <vector>
#include <bit>

template <typename KernelStruct>
concept SerializableKernelArgs = alignof(KernelStruct) == alignof(uint32_t) && std::is_aggregate_v<KernelStruct> &&
                                 std::is_trivially_copyable_v<KernelStruct>;

template <SerializableKernelArgs KernStruct>
consteval uint32_t amount_of_fields() {
    return sizeof(KernStruct) / sizeof(uint32_t);
}

template <SerializableKernelArgs KernStruct>
std::vector<uint32_t> to_vector(const KernStruct& kernel_data) {
    constexpr auto N = amount_of_fields<KernStruct>();
    auto res = std::bit_cast<std::array<std::uint32_t, N>>(kernel_data);
    return std::vector<uint32_t>(res.begin(), res.end());
}

#else
template <typename KernelStruct>
inline constexpr bool SerializableKernelArgs =
    alignof(KernelStruct) == alignof(uint32_t) && std::is_aggregate_v<KernelStruct> &&
    std::is_trivially_copyable_v<KernelStruct>;
#endif
}  // namespace ttnn::kernel_utils
