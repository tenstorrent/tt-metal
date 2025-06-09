#pragma once

#if !defined(KERNEL_BUILD)

#include <type_traits>
#include <reflect>
#include <vector>

namespace ttnn::kernel_utils {

template <typename KernelStruct, typename FieldType = uint32_t>
concept KernelArgsStructConcept =
    alignof(KernelStruct) == alignof(FieldType) && sizeof(KernelStruct) % sizeof(FieldType) == 0 &&
    std::is_aggregate_v<KernelStruct> && [] {
        bool result = true;
        KernelStruct val{};
        reflect::for_each(
            [&](auto I) { result &= reflect::type_name(FieldType{}) == reflect::type_name(reflect::get<I>(val)); },
            val);
        return result;
    }();

template <KernelArgsStructConcept KernStruct>
std::vector<uint32_t> to_vector(const KernStruct& kernel_data) {
    std::vector<uint32_t> res(sizeof(KernStruct) / sizeof(uint32_t), 0);
    uint32_t idx = 0;
    reflect::for_each([&](auto I) { res[idx++] = reflect::get<I>(kernel_data); }, kernel_data);
    return res;
}

template <KernelArgsStructConcept KernStruct>
consteval uint32_t amount_of_fields() {
    return sizeof(KernStruct) / sizeof(reflect::get<0>(KernStruct{}));
}

}  // namespace ttnn::kernel_utils

#define VALIDATE_KERNEL_ARGS_STRUCT(KernStruct)                  \
    static_assert(                                               \
        ttnn::kernel_utils::KernelArgsStructConcept<KernStruct>, \
        "Struct does not satisfy the requirements of KernelArgsStructConcept.");

#else
#define VALIDATE_KERNEL_ARGS_STRUCT(KernStruct)
#endif
