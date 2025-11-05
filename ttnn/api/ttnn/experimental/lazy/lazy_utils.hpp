// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <optional>
#include <functional>
#include <tuple>
#include <utility>

namespace ttnn::experimental::lazy {

template <typename object_t, typename T>
std::vector<T> object_to_vector(const object_t& object) {
    std::vector<T> vector;
    tt::stl::reflection::visit_object_of_type<T>([&](const auto& t) { vector.push_back(t); }, object);
    return vector;
}

// Helper to collect tensor structure metadata (vector sizes, optional presence, etc.)
struct TensorArgsMetadata {
    std::vector<size_t> vector_sizes;     // Size of each vector<Tensor> or vector<optional<Tensor>> encountered
    std::vector<bool> optional_presence;  // Whether each optional<Tensor> has value
    size_t total_tensor_count = 0;        // Total number of tensors in the structure
};

// Helper to collect metadata from tensor_args
template <typename tensor_args_t>
TensorArgsMetadata collect_tensor_args_meta(const tensor_args_t& tensor_args) {
    TensorArgsMetadata metadata;

    // We need to traverse the structure manually to capture vector sizes and optional presence
    collect_tensor_structure_impl(metadata, tensor_args);

    return metadata;
}

// Implementation detail - recursively collect structure info
template <typename T>
void collect_tensor_structure_impl(TensorArgsMetadata& metadata, const T& value) {
    if constexpr (std::same_as<std::decay_t<T>, Tensor>) {
        metadata.total_tensor_count++;
    } else if constexpr (requires { value.has_value(); }) {
        // std::optional<Tensor> or std::optional<std::vector<Tensor>>
        bool has_value = value.has_value();
        metadata.optional_presence.push_back(has_value);
        if (has_value) {
            collect_tensor_structure_impl(metadata, *value);
        }
    } else if constexpr (requires {
                             value.size();
                             value.begin();
                             value.end();
                         }) {
        // std::vector<Tensor> or std::vector<std::optional<Tensor>>
        size_t size = value.size();
        metadata.vector_sizes.push_back(size);
        for (const auto& elem : value) {
            collect_tensor_structure_impl(metadata, elem);
        }
    } else if constexpr (requires { std::tuple_size<std::decay_t<T>>::value; }) {
        // Tuple or similar
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (collect_tensor_structure_impl(metadata, std::get<Is>(value)), ...);
        }(std::make_index_sequence<std::tuple_size_v<std::decay_t<T>>>{});
    } else if constexpr (ttsl::concepts::Reflectable<std::decay_t<T>>) {
        // Reflectable struct
        reflect::for_each([&](auto I) { collect_tensor_structure_impl(metadata, reflect::get<I>(value)); }, value);
    } else if constexpr (requires { std::decay_t<T>::attribute_names; }) {
        // Struct with attribute_names
        collect_tensor_structure_impl(metadata, value.attribute_values());
    }
}

// Helper to reconstruct tensor_args from flat input tensor list
struct TensorReconstructionContext {
    const std::vector<tt::tt_metal::metal_tensor::Tensor>& input_tensors;
    const TensorArgsMetadata& metadata;
    size_t tensor_index = 0;
    size_t vector_size_index = 0;
    size_t optional_index = 0;

    TensorReconstructionContext(
        const std::vector<tt::tt_metal::metal_tensor::Tensor>& tensors, const TensorArgsMetadata& meta) :
        input_tensors(tensors), metadata(meta) {}
};

// Forward declare reconstruction function
template <typename T>
T reconstruct_value(TensorReconstructionContext& ctx);

// Helper to check if a type is Tensor (handling const)
template <typename T>
concept IsTensor = std::same_as<std::remove_const_t<T>, Tensor>;

// Specialization for Tensor and const Tensor
template <IsTensor T>
inline T reconstruct_value(TensorReconstructionContext& ctx) {
    TT_FATAL(ctx.tensor_index < ctx.input_tensors.size(), "Tensor index out of bounds during reconstruction");
    // Make an explicit copy of the metal tensor to ensure proper ownership
    auto metal_tensor_copy = ctx.input_tensors[ctx.tensor_index++];
    return Tensor(metal_tensor_copy);
}

// Specialization for std::optional<Tensor> and std::optional<const Tensor>
template <typename T>
    requires requires { typename T::value_type; } && IsTensor<typename T::value_type> &&
             requires(T t) { t.has_value(); }
inline T reconstruct_value(TensorReconstructionContext& ctx) {
    TT_FATAL(ctx.optional_index < ctx.metadata.optional_presence.size(), "Optional index out of bounds");
    bool has_value = ctx.metadata.optional_presence[ctx.optional_index++];
    if (has_value) {
        TT_FATAL(ctx.tensor_index < ctx.input_tensors.size(), "Tensor index out of bounds during reconstruction");
        // Make an explicit copy of the metal tensor to ensure proper ownership
        auto metal_tensor_copy = ctx.input_tensors[ctx.tensor_index++];
        return T(Tensor(metal_tensor_copy));
    }
    return std::nullopt;
}

// Specialization for std::vector<Tensor> and std::vector<const Tensor>
template <typename T>
    requires requires { typename T::value_type; } && IsTensor<typename T::value_type> &&
             requires(T t) {
                 t.size();
                 t.begin();
                 t.end();
             } && (!requires(typename T::value_type v) { v.has_value(); })  // Not optional
inline T reconstruct_value(TensorReconstructionContext& ctx) {
    TT_FATAL(ctx.vector_size_index < ctx.metadata.vector_sizes.size(), "Vector size index out of bounds");
    size_t size = ctx.metadata.vector_sizes[ctx.vector_size_index++];
    T result;
    result.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        TT_FATAL(ctx.tensor_index < ctx.input_tensors.size(), "Tensor index out of bounds during reconstruction");
        // Make an explicit copy of the metal tensor to ensure proper ownership
        auto metal_tensor_copy = ctx.input_tensors[ctx.tensor_index++];
        result.emplace_back(metal_tensor_copy);
    }
    return result;
}

// Specialization for std::vector<std::optional<Tensor>> and std::vector<std::optional<const Tensor>>
template <typename T>
    requires requires { typename T::value_type::value_type; } && IsTensor<typename T::value_type::value_type> &&
             requires(T t) {
                 t.size();
                 t.begin();
                 t.end();
             } && requires(typename T::value_type v) { v.has_value(); }  // Is vector of optionals
inline T reconstruct_value(TensorReconstructionContext& ctx) {
    TT_FATAL(ctx.vector_size_index < ctx.metadata.vector_sizes.size(), "Vector size index out of bounds");
    size_t size = ctx.metadata.vector_sizes[ctx.vector_size_index++];
    T result;
    result.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        TT_FATAL(ctx.optional_index < ctx.metadata.optional_presence.size(), "Optional index out of bounds");
        bool has_value = ctx.metadata.optional_presence[ctx.optional_index++];
        if (has_value) {
            TT_FATAL(ctx.tensor_index < ctx.input_tensors.size(), "Tensor index out of bounds during reconstruction");
            // Make an explicit copy of the metal tensor to ensure proper ownership
            auto metal_tensor_copy = ctx.input_tensors[ctx.tensor_index++];
            result.emplace_back(metal_tensor_copy);
        } else {
            result.emplace_back(std::nullopt);
        }
    }
    return result;
}

// Generic fallback for reflectable structs (like tensor_args_t)
template <typename T>
    requires ttsl::concepts::Reflectable<std::decay_t<T>> && (!IsTensor<T>) &&
             (!requires { typename T::value_type; })  // Not a container
inline T reconstruct_value(TensorReconstructionContext& ctx) {
    return reconstruct_tensor_args_impl<T>(ctx);
}

// Generic reconstruction for reflectable structs
template <typename T>
    requires ttsl::concepts::Reflectable<std::decay_t<T>>
T reconstruct_tensor_args_impl(TensorReconstructionContext& ctx) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return T{reconstruct_value<std::decay_t<decltype(reflect::get<Is>(std::declval<T>()))>>(ctx)...};
    }(std::make_index_sequence<reflect::size<T>()>{});
}

}  // namespace ttnn::experimental::lazy
