// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <array>
#include <tuple>

namespace ttnn::experimental::lazy {

// Type trait helpers
template <typename T>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

// Check if type is tuple-like (tuple or array)
template <typename T>
concept TupleLike = requires { std::tuple_size<std::decay_t<T>>::value; };

// Check if type is range-like (has begin/end)
template <typename T>
concept RangeLike = requires(T t) {
    { t.begin() } -> std::input_or_output_iterator;
    { t.end() } -> std::input_or_output_iterator;
};

// Helper to convert any spec-like type to optional<TensorSpec>
template <typename T>
std::optional<TensorSpec> to_optional_spec(const T& value) {
    using value_t = std::decay_t<T>;
    if constexpr (std::same_as<value_t, TensorSpec>) {
        return value;
    } else if constexpr (std::same_as<value_t, std::optional<TensorSpec>>) {
        return value;
    } else {
        static_assert(std::same_as<value_t, void>, "Unsupported spec element type");
        return std::nullopt;
    }
}

// Flatten any spec type to vector<optional<TensorSpec>>
template <typename SpecType>
std::vector<std::optional<TensorSpec>> flatten_specs(const SpecType& specs) {
    using spec_t = std::decay_t<SpecType>;

    // Single element (TensorSpec or optional<TensorSpec>)
    if constexpr (std::same_as<spec_t, TensorSpec> || std::same_as<spec_t, std::optional<TensorSpec>>) {
        return {to_optional_spec(specs)};
    }
    // Tuple-like (array or tuple) - convert to vector
    else if constexpr (TupleLike<spec_t>) {
        constexpr auto N = std::tuple_size_v<spec_t>;
        std::vector<std::optional<TensorSpec>> result;
        result.reserve(N);

        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (result.push_back(to_optional_spec(std::get<Is>(specs))), ...);
        }(std::make_index_sequence<N>{});

        return result;
    }
    // Range-like (vector, etc) - iterate and convert
    else if constexpr (RangeLike<spec_t>) {
        std::vector<std::optional<TensorSpec>> result;
        for (const auto& elem : specs) {
            result.push_back(to_optional_spec(elem));
        }
        return result;
    } else {
        static_assert(std::same_as<spec_t, void>, "Unsupported spec type");
        return {};
    }
}

// Helper to get element type from iterable or tuple-like type
template <typename T>
struct element_type {
    using type = void;
};

// TupleLike types (array, tuple)
template <TupleLike T>
struct element_type<T> {
    using type = std::decay_t<std::tuple_element_t<0, T>>;
};

// RangeLike types (vector, etc) - exclude TupleLike to avoid ambiguity with std::array
template <typename T>
    requires(RangeLike<T> && !TupleLike<T>)
struct element_type<T> {
    using type = std::decay_t<decltype(*std::declval<T>().begin())>;
};

template <typename T>
using element_type_t = typename element_type<T>::type;

// Reconstruct tensor_return_value_t from vector<Tensor>
template <typename ReturnType>
ReturnType reconstruct_return_value(
    const std::vector<Tensor>& tensors, const std::vector<std::optional<TensorSpec>>& output_specs) {
    using return_t = std::decay_t<ReturnType>;

    // Single element (Tensor or optional<Tensor>)
    if constexpr (std::same_as<return_t, Tensor>) {
        TT_FATAL(tensors.size() == 1, "Expected 1 tensor");
        return tensors[0];
    } else if constexpr (std::same_as<return_t, std::optional<Tensor>>) {
        TT_FATAL(output_specs.size() == 1, "Expected 1 spec");
        if (!output_specs[0].has_value()) {
            return std::nullopt;
        }
        TT_FATAL(tensors.size() == 1, "Expected 1 tensor");
        return tensors[0];
    }
    // Tuple-like (array or tuple) - convert from vector
    else if constexpr (TupleLike<return_t>) {
        constexpr auto N = std::tuple_size_v<return_t>;
        using elem_t = element_type_t<return_t>;

        if constexpr (is_optional_v<elem_t>) {
            // Tuple of optional<Tensor>
            TT_FATAL(output_specs.size() == N, "Expected {} specs", N);
            size_t tensor_idx = 0;
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return return_t{
                    (output_specs[Is].has_value() ? std::optional<Tensor>{tensors[tensor_idx++]} : std::nullopt)...};
            }(std::make_index_sequence<N>{});
        } else {
            // Tuple of Tensor
            TT_FATAL(tensors.size() == N, "Expected {} tensors", N);
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return return_t{tensors[Is]...};
            }(std::make_index_sequence<N>{});
        }
    }
    // Range-like (vector, etc) - convert from vector
    else if constexpr (RangeLike<return_t>) {
        using elem_t = element_type_t<return_t>;

        if constexpr (is_optional_v<elem_t>) {
            // Vector of optional<Tensor> - reconstruct with nullopts
            return_t result;
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
        } else {
            // Vector of Tensor - return as-is
            return tensors;
        }
    } else {
        static_assert(
            std::same_as<return_t, void>,
            "Unsupported return type. Supported return types are Tensor, optional<Tensor>, tuple<Tensor, ...>, "
            "array<Tensor, ...>, vector<Tensor> and vector<optional<Tensor>>.");
    }
}

}  // namespace ttnn::experimental::lazy
