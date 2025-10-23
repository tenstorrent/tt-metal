// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <boost/pfr.hpp>
#include <optional>
#include <functional>
#include <tuple>
#include <utility>

namespace ttnn::experimental::jit {

// ============================================================================
// from_range_pfr: Construct aggregate structs from iterator ranges
//
// This solves the problem of constructing structs with reference members from
// vectors. References must be bound during construction, not after (which is UB).
// ============================================================================

// Helper to get the Nth field type of a struct S
// IMPORTANT: Use S& (lvalue reference) to preserve reference field types
template <typename S, std::size_t N>
struct field_type {
    using type = decltype(boost::pfr::get<N>(std::declval<S&>()));
};

template <typename S, std::size_t N>
using field_type_t = typename field_type<S, N>::type;

// Storage type for a field - either the value itself or a pointer (for Tensor references only)
template <typename FieldType>
struct field_storage {
    using BareType = std::remove_cv_t<std::remove_reference_t<FieldType>>;

    // Only store as pointer if it's a reference to Tensor (not optional<Tensor> etc.)
    using type = std::conditional_t<
        std::is_reference_v<FieldType> && std::is_same_v<BareType, Tensor>,
        std::remove_reference_t<FieldType>*,  // Store as pointer
        std::remove_reference_t<FieldType>    // Store by value (removes the & from pfr::get)
        >;
};

template <typename FieldType>
using field_storage_t = typename field_storage<FieldType>::type;

// Helper to extract a field from storage (dereference if it's a pointer, otherwise return ref to value)
template <typename FieldType, typename StorageType>
decltype(auto) extract_from_storage(StorageType& storage) {
    using BareFieldType = std::remove_cv_t<std::remove_reference_t<FieldType>>;

    // If FieldType is a Tensor reference, storage is a pointer - dereference it
    if constexpr (std::is_reference_v<FieldType> && std::is_same_v<BareFieldType, Tensor>) {
        return *storage;  // Dereference pointer to get lvalue reference
    } else {
        // Otherwise storage holds the value, return it (by ref if needed)
        return static_cast<FieldType>(storage);
    }
}

// Helper to bind a single field
template <typename S, std::size_t N, typename It>
static field_storage_t<field_type_t<S, N>> bind_field_to_storage(It& it, It end) {
    using FieldType = field_type_t<S, N>;
    using BareFieldType = std::remove_cv_t<std::remove_reference_t<FieldType>>;

    // Check if this is a Tensor reference (not optional<Tensor>& etc.)
    if constexpr (std::is_reference_v<FieldType> && std::is_same_v<BareFieldType, Tensor>) {
        if (it == end) {
            TT_THROW("Not enough tensors for reference field");
        }
        Tensor& r = *it;
        ++it;
        // Return pointer to the element
        return &r;
    } else if constexpr (std::is_same_v<BareFieldType, Tensor>) {
        // Plain Tensor by value
        if (it == end) {
            TT_THROW("Not enough tensors for value field");
        }
        Tensor v = *it;
        ++it;
        return v;
    } else if constexpr (std::is_same_v<BareFieldType, std::optional<Tensor>>) {
        if (it == end) {
            return std::optional<Tensor>(std::nullopt);
        }
        Tensor v = *it;
        ++it;
        return std::optional<Tensor>(std::move(v));
    } else if constexpr (std::is_same_v<BareFieldType, std::optional<std::reference_wrapper<const Tensor>>>) {
        if (it == end) {
            return std::optional<std::reference_wrapper<const Tensor>>(std::nullopt);
        }
        Tensor& r = *it;
        ++it;
        return std::optional<std::reference_wrapper<const Tensor>>(std::cref(r));
    } else if constexpr (std::is_same_v<BareFieldType, std::optional<std::reference_wrapper<Tensor>>>) {
        if (it == end) {
            return std::optional<std::reference_wrapper<Tensor>>(std::nullopt);
        }
        Tensor& r = *it;
        ++it;
        return std::optional<std::reference_wrapper<Tensor>>(std::ref(r));
    } else {
        static_assert(std::is_same_v<FieldType, void>, "Unsupported field wrapper type");
        return field_storage_t<FieldType>{};  // unreachable
    }
}

// Helper to build storage tuple
template <typename S, typename It, std::size_t... Indices>
static auto build_storage_tuple(It& it, It end, std::index_sequence<Indices...>) {
    return std::tuple<field_storage_t<field_type_t<S, Indices>>...>{bind_field_to_storage<S, Indices>(it, end)...};
}

// Helper to construct struct from storage - specialized for different field counts
// We need explicit specializations because aggregate initialization requires actual lvalue expressions

template <typename S, typename StorageTuple, std::size_t N>
struct construct_from_storage_impl;

// Specialization for 1 field
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 1> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        return S{f0};
    }
};

// Specialization for 2 fields
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 2> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        decltype(auto) f1 = extract_from_storage<field_type_t<S, 1>>(std::get<1>(storage));
        return S{f0, f1};
    }
};

// Specialization for 3 fields
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 3> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        decltype(auto) f1 = extract_from_storage<field_type_t<S, 1>>(std::get<1>(storage));
        decltype(auto) f2 = extract_from_storage<field_type_t<S, 2>>(std::get<2>(storage));
        return S{f0, f1, f2};
    }
};

// Specialization for 4 fields
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 4> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        decltype(auto) f1 = extract_from_storage<field_type_t<S, 1>>(std::get<1>(storage));
        decltype(auto) f2 = extract_from_storage<field_type_t<S, 2>>(std::get<2>(storage));
        decltype(auto) f3 = extract_from_storage<field_type_t<S, 3>>(std::get<3>(storage));
        return S{f0, f1, f2, f3};
    }
};

// Specialization for 5 fields
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 5> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        decltype(auto) f1 = extract_from_storage<field_type_t<S, 1>>(std::get<1>(storage));
        decltype(auto) f2 = extract_from_storage<field_type_t<S, 2>>(std::get<2>(storage));
        decltype(auto) f3 = extract_from_storage<field_type_t<S, 3>>(std::get<3>(storage));
        decltype(auto) f4 = extract_from_storage<field_type_t<S, 4>>(std::get<4>(storage));
        return S{f0, f1, f2, f3, f4};
    }
};

// Specialization for 6 fields
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 6> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        decltype(auto) f1 = extract_from_storage<field_type_t<S, 1>>(std::get<1>(storage));
        decltype(auto) f2 = extract_from_storage<field_type_t<S, 2>>(std::get<2>(storage));
        decltype(auto) f3 = extract_from_storage<field_type_t<S, 3>>(std::get<3>(storage));
        decltype(auto) f4 = extract_from_storage<field_type_t<S, 4>>(std::get<4>(storage));
        decltype(auto) f5 = extract_from_storage<field_type_t<S, 5>>(std::get<5>(storage));
        return S{f0, f1, f2, f3, f4, f5};
    }
};

// Specialization for 7 fields
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 7> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        decltype(auto) f1 = extract_from_storage<field_type_t<S, 1>>(std::get<1>(storage));
        decltype(auto) f2 = extract_from_storage<field_type_t<S, 2>>(std::get<2>(storage));
        decltype(auto) f3 = extract_from_storage<field_type_t<S, 3>>(std::get<3>(storage));
        decltype(auto) f4 = extract_from_storage<field_type_t<S, 4>>(std::get<4>(storage));
        decltype(auto) f5 = extract_from_storage<field_type_t<S, 5>>(std::get<5>(storage));
        decltype(auto) f6 = extract_from_storage<field_type_t<S, 6>>(std::get<6>(storage));
        return S{f0, f1, f2, f3, f4, f5, f6};
    }
};

// Specialization for 8 fields
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 8> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        decltype(auto) f1 = extract_from_storage<field_type_t<S, 1>>(std::get<1>(storage));
        decltype(auto) f2 = extract_from_storage<field_type_t<S, 2>>(std::get<2>(storage));
        decltype(auto) f3 = extract_from_storage<field_type_t<S, 3>>(std::get<3>(storage));
        decltype(auto) f4 = extract_from_storage<field_type_t<S, 4>>(std::get<4>(storage));
        decltype(auto) f5 = extract_from_storage<field_type_t<S, 5>>(std::get<5>(storage));
        decltype(auto) f6 = extract_from_storage<field_type_t<S, 6>>(std::get<6>(storage));
        decltype(auto) f7 = extract_from_storage<field_type_t<S, 7>>(std::get<7>(storage));
        return S{f0, f1, f2, f3, f4, f5, f6, f7};
    }
};

// Specialization for 9 fields
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 9> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        decltype(auto) f1 = extract_from_storage<field_type_t<S, 1>>(std::get<1>(storage));
        decltype(auto) f2 = extract_from_storage<field_type_t<S, 2>>(std::get<2>(storage));
        decltype(auto) f3 = extract_from_storage<field_type_t<S, 3>>(std::get<3>(storage));
        decltype(auto) f4 = extract_from_storage<field_type_t<S, 4>>(std::get<4>(storage));
        decltype(auto) f5 = extract_from_storage<field_type_t<S, 5>>(std::get<5>(storage));
        decltype(auto) f6 = extract_from_storage<field_type_t<S, 6>>(std::get<6>(storage));
        decltype(auto) f7 = extract_from_storage<field_type_t<S, 7>>(std::get<7>(storage));
        decltype(auto) f8 = extract_from_storage<field_type_t<S, 8>>(std::get<8>(storage));
        return S{f0, f1, f2, f3, f4, f5, f6, f7, f8};
    }
};

// Specialization for 10 fields
template <typename S, typename StorageTuple>
struct construct_from_storage_impl<S, StorageTuple, 10> {
    static S construct(StorageTuple& storage) {
        decltype(auto) f0 = extract_from_storage<field_type_t<S, 0>>(std::get<0>(storage));
        decltype(auto) f1 = extract_from_storage<field_type_t<S, 1>>(std::get<1>(storage));
        decltype(auto) f2 = extract_from_storage<field_type_t<S, 2>>(std::get<2>(storage));
        decltype(auto) f3 = extract_from_storage<field_type_t<S, 3>>(std::get<3>(storage));
        decltype(auto) f4 = extract_from_storage<field_type_t<S, 4>>(std::get<4>(storage));
        decltype(auto) f5 = extract_from_storage<field_type_t<S, 5>>(std::get<5>(storage));
        decltype(auto) f6 = extract_from_storage<field_type_t<S, 6>>(std::get<6>(storage));
        decltype(auto) f7 = extract_from_storage<field_type_t<S, 7>>(std::get<7>(storage));
        decltype(auto) f8 = extract_from_storage<field_type_t<S, 8>>(std::get<8>(storage));
        decltype(auto) f9 = extract_from_storage<field_type_t<S, 9>>(std::get<9>(storage));
        return S{f0, f1, f2, f3, f4, f5, f6, f7, f8, f9};
    }
};

// Wrapper that dispatches based on number of fields
template <typename S, typename StorageTuple, std::size_t... Indices>
static S construct_from_storage(StorageTuple& storage, std::index_sequence<Indices...>) {
    constexpr std::size_t N = sizeof...(Indices);
    static_assert(N >= 1 && N <= 10, "Struct must have between 1 and 10 fields. Add more specializations if needed.");
    return construct_from_storage_impl<S, StorageTuple, N>::construct(storage);
}

// Main function: construct struct S from iterator range
template <typename S, typename It>
static S from_range_impl(It it, It end) {
    constexpr std::size_t num_fields = boost::pfr::tuple_size_v<S>;

    // Build storage tuple with all field values/references
    auto storage = build_storage_tuple<S>(it, end, std::make_index_sequence<num_fields>{});

    // Extract from storage and construct the struct
    return construct_from_storage<S>(storage, std::make_index_sequence<num_fields>{});
}

// Public API: construct struct S from iterator range
template <typename S, typename It>
S from_range_pfr(It it, It end) {
    return from_range_impl<S>(it, end);
}

}  // namespace ttnn::experimental::jit
