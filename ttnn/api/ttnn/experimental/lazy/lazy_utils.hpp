// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <array>
#include <tuple>

namespace ttnn::experimental::lazy {

// Check if type is std::array<Tensor, N>
template <typename T>
struct is_tensor_array : std::false_type {};

template <std::size_t N>
struct is_tensor_array<std::array<Tensor, N>> : std::true_type {};

template <typename T>
inline constexpr bool is_tensor_array_v = is_tensor_array<T>::value;

// Check if type is std::tuple of all Tensors
template <typename T>
struct is_all_tensor_tuple : std::false_type {};

template <typename... Types>
struct is_all_tensor_tuple<std::tuple<Types...>> : std::bool_constant<(std::same_as<Types, Tensor> && ...)> {};

template <typename T>
inline constexpr bool is_all_tensor_tuple_v = is_all_tensor_tuple<T>::value;

// Flatten any spec type to vector<optional<TensorSpec>>
template <typename SpecType>
std::vector<std::optional<TensorSpec>> flatten_specs(const SpecType& specs) {
    using spec_t = std::decay_t<SpecType>;

    if constexpr (std::same_as<spec_t, TensorSpec>) {
        // Single spec
        return {specs};
    } else if constexpr (std::same_as<spec_t, std::vector<TensorSpec>>) {
        // Vector of specs (all non-optional)
        std::vector<std::optional<TensorSpec>> result;
        result.reserve(specs.size());
        for (const auto& spec : specs) {
            result.push_back(spec);
        }
        return result;
    } else if constexpr (std::same_as<spec_t, std::vector<std::optional<TensorSpec>>>) {
        // Vector of optional specs - return as-is
        return specs;
    } else if constexpr (requires { std::tuple_size<spec_t>::value; }) {
        // Tuple (array or tuple) - convert to vector
        constexpr auto N = std::tuple_size_v<spec_t>;
        std::vector<std::optional<TensorSpec>> result;
        result.reserve(N);

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (
                (void)[&] {
                    const auto& elem = std::get<Is>(specs);
                    using elem_t = std::decay_t<decltype(elem)>;
                    if constexpr (std::same_as<elem_t, TensorSpec>) {
                        result.push_back(elem);
                    } else if constexpr (std::same_as<elem_t, std::optional<TensorSpec>>) {
                        result.push_back(elem);
                    }
                }(),
                ...);
        }(std::make_index_sequence<N>{});

        return result;
    } else {
        static_assert(std::same_as<spec_t, void>, "Unsupported spec type");
        return {};
    }
}

// Reconstruct tensor_return_value_t from vector<Tensor>
template <typename ReturnType>
ReturnType reconstruct_return_value(
    const std::vector<Tensor>& tensors, const std::vector<std::optional<TensorSpec>>& output_specs) {
    using return_t = std::decay_t<ReturnType>;

    if constexpr (std::same_as<return_t, Tensor>) {
        // Single tensor
        TT_FATAL(tensors.size() == 1, "Expected 1 tensor");
        return tensors[0];
    } else if constexpr (std::same_as<return_t, std::vector<Tensor>>) {
        // Vector of tensors
        return tensors;
    } else if constexpr (std::same_as<return_t, std::vector<std::optional<Tensor>>>) {
        // Vector of optional tensors - reconstruct with nullopts
        std::vector<std::optional<Tensor>> result;
        result.reserve(output_specs.size());
        size_t tensor_idx = 0;
        for (const auto& spec_opt : output_specs) {
            if (!spec_opt.has_value()) {
                result.push_back(std::nullopt);
            } else {
                result.push_back(tensors[tensor_idx++]);
            }
        }
        return result;
    } else if constexpr (is_tensor_array_v<return_t>) {
        // std::array<Tensor, N>
        constexpr auto N = std::tuple_size_v<return_t>;
        TT_FATAL(tensors.size() == N, "Expected {} tensors", N);
        return_t result;
        for (size_t i = 0; i < N; i++) {
            result[i] = tensors[i];
        }
        return result;
    } else if constexpr (is_all_tensor_tuple_v<return_t>) {
        // std::tuple<Tensor, ...>
        constexpr auto N = std::tuple_size_v<return_t>;
        TT_FATAL(tensors.size() == N, "Expected {} tensors", N);
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return return_t{tensors[Is]...};
        }(std::make_index_sequence<N>{});
    } else {
        static_assert(std::same_as<return_t, void>, "Unsupported return type");
    }
}

}  // namespace ttnn::experimental::lazy
