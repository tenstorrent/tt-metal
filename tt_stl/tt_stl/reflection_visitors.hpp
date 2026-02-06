// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Visitor / transform / get_first_object_of_type utilities.
// Split out of reflection.hpp so that the ~415 lines of deeply-nested
// template code are only parsed by the handful of translation units that
// actually need them (device_operation, op_profiler, graph queries, …).

#pragma once

#include <tt_stl/reflection.hpp>

namespace ttsl::reflection {

template <typename T>
struct visit_object_of_type_t;

template <typename object_t, typename T>
void visit_object_of_type(auto&& callback, T&& object) {
    visit_object_of_type_t<std::decay_t<T>>{}.template operator()<object_t>(
        std::forward<decltype(callback)>(callback), std::forward<T>(object));
}

template <typename T>
    requires(not ttsl::concepts::Reflectable<std::decay_t<T>>) and (not requires { std::decay_t<T>::attribute_names; })
struct visit_object_of_type_t<T> {
    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, T&& value) const {
        callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& /*callback*/, T&& /*value*/) const {
        throw std::runtime_error(fmt::format("Unsupported visit of object of type: {}", get_type_name<T>()));
    }

    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, const T& value) const {
        callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& /*callback*/, const T& /*value*/) const {
        throw std::runtime_error(fmt::format("Unsupported visit of object of type: {}", get_type_name<T>()));
    }
};

template <typename T>
struct visit_object_of_type_t<std::optional<T>> {
    template <typename object_t>
    void operator()(auto&& callback, const std::optional<T>& value) const {
        if (value.has_value()) {
            visit_object_of_type<object_t>(callback, value.value());
        }
    }
};

template <typename T>
struct visit_object_of_type_t<std::vector<T>> {
    template <typename object_t>
    void operator()(auto&& callback, const std::vector<T>& value) const {
        for (auto& tensor : value) {
            visit_object_of_type<object_t>(callback, tensor);
        }
    }
};

template <typename T, auto N>
struct visit_object_of_type_t<std::array<T, N>> {
    template <typename object_t>
    void operator()(auto&& callback, const std::array<T, N>& value) const {
        for (auto& tensor : value) {
            visit_object_of_type<object_t>(callback, tensor);
        }
    }
};

template <typename... Ts>
struct visit_object_of_type_t<std::tuple<Ts...>> {
    template <typename object_t>
    void operator()(auto&& callback, const std::tuple<Ts...>& value) const {
        [&callback, &value]<size_t... Ns>(std::index_sequence<Ns...>) {
            (visit_object_of_type<object_t>(callback, std::get<Ns>(value)), ...);
        }(std::make_index_sequence<sizeof...(Ts)>{});
    }
};

template <typename T>
    requires requires { std::decay_t<T>::attribute_names; }
struct visit_object_of_type_t<T> {
    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, T&& value) const {
        callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, T&& object) const {
        constexpr auto num_attributes = std::tuple_size_v<decltype(std::decay_t<T>::attribute_names)>;
        visit_object_of_type<object_t>(callback, object.attribute_values());
    }

    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, const T& value) const {
        callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, const T& object) const {
        constexpr auto num_attributes = std::tuple_size_v<decltype(std::decay_t<T>::attribute_names)>;
        visit_object_of_type<object_t>(callback, object.attribute_values());
    }
};

template <typename T>
    requires ttsl::concepts::Reflectable<std::decay_t<T>>
struct visit_object_of_type_t<T> {
    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, T&& value) const {
        callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, T&& object) const {
        reflect::for_each(
            [&callback, &object](auto I) { visit_object_of_type<object_t>(callback, reflect::get<I>(object)); },
            object);
    }

    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, const T& value) const {
        callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, const T& object) const {
        reflect::for_each(
            [&callback, &object](auto I) { visit_object_of_type<object_t>(callback, reflect::get<I>(object)); },
            object);
    }
};

template <typename T>
struct transform_object_of_type_t;

template <typename object_t, typename T>
auto transform_object_of_type(auto&& callback, T&& object) {
    return transform_object_of_type_t<std::decay_t<T>>{}.template operator()<object_t>(
        std::forward<decltype(callback)>(callback), std::forward<T>(object));
}

template <typename T>
    requires(not ttsl::concepts::Reflectable<std::decay_t<T>>) and (not requires { std::decay_t<T>::attribute_names; })
struct transform_object_of_type_t<T> {
    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, T&& value) const {
        return callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& /*callback*/, T&& value) const {
        log_debug(tt::LogAlways, "Unsupported transform of object of type: {}. Do nothing.", get_type_name<T>());
        return value;
    }

    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, const T& value) const {
        return callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& /*callback*/, const T& value) const {
        log_debug(tt::LogAlways, "Unsupported transform of object of type: {}. Do nothing.", get_type_name<T>());
        return value;
    }
};

template <typename T>
struct transform_object_of_type_t<std::optional<T>> {
    template <typename object_t>
    std::optional<T> operator()(auto&& callback, const std::optional<T>& value) const {
        if (value.has_value()) {
            return transform_object_of_type<object_t>(callback, value.value());
        }
        return std::nullopt;
    }
};

template <typename T>
struct transform_object_of_type_t<std::vector<T>> {
    template <typename object_t>
    std::vector<T> operator()(auto&& callback, const std::vector<T>& value) const {
        std::vector<T> return_value;
        return_value.reserve(value.size());
        for (auto& tensor : value) {
            return_value.emplace_back(transform_object_of_type<object_t>(callback, tensor));
        }
        return return_value;
    }
};

template <typename T, auto N>
struct transform_object_of_type_t<std::array<T, N>> {
    template <typename object_t>
    std::array<T, N> operator()(auto&& callback, const std::array<T, N>& value) const {
        std::array<T, N> return_value;
        for (auto index = 0; index < value.size(); index++) {
            return_value[index] = transform_object_of_type<object_t>(callback, value[index]);
        }
        return return_value;
    }
};

template <typename... Ts>
struct transform_object_of_type_t<std::tuple<Ts...>> {
    template <typename object_t>
    std::tuple<Ts...> operator()(auto&& callback, const std::tuple<Ts...>& value) const {
        return std::apply(
            [&callback]<typename... args_t>(args_t&&... args) {
                return std::make_tuple(transform_object_of_type<object_t>(callback, std::forward<args_t>(args))...);
            },
            value);
    }
};

template <typename T>
    requires requires { std::decay_t<T>::attribute_names; }
struct transform_object_of_type_t<T> {
    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, T&& value) const {
        return callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& /*callback*/, T&& /*object*/) const {
        static_assert(ttsl::concepts::always_false_v<T>, "Unsupported transform of object of type");
    }

    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, const T& value) const {
        return callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& /*callback*/, const T& /*object*/) const {
        static_assert(ttsl::concepts::always_false_v<T>, "Unsupported transform of object of type");
    }
};

template <typename T>
    requires ttsl::concepts::Reflectable<std::decay_t<T>>
struct transform_object_of_type_t<T> {
    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, T&& value) const {
        return callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& callback, T&& object) const {
        return std::apply(
            [&callback](auto&&... args) {
                return T{transform_object_of_type<object_t>(callback, std::forward<decltype(args)>(args))...};
            },
            reflect::to<std::tuple>(object));
    }

    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, const T& value) const {
        return callback(value);
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& callback, const T& object) const {
        return std::apply(
            [&callback](auto&&... args) {
                return T{transform_object_of_type<object_t>(callback, std::forward<decltype(args)>(args))...};
            },
            reflect::to<std::tuple>(object));
    }
};

template <typename T>
struct get_first_object_of_type_t;

template <typename object_t, typename T>
auto get_first_object_of_type(const T& value) {
    return get_first_object_of_type_t<std::decay_t<T>>{}.template operator()<object_t>(value);
}

template <typename T>
    requires(not ttsl::concepts::Reflectable<std::decay_t<T>>) and (not requires { std::decay_t<T>::attribute_names; })
struct get_first_object_of_type_t<T> {
    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    auto operator()(const T& value) const -> std::optional<object_t> {
        return value;
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    auto operator()(const T& /*value*/) const -> std::optional<object_t> {
        return std::nullopt;
    }
};

template <typename T>
struct get_first_object_of_type_t<std::optional<T>> {
    template <typename object_t>
    auto operator()(const std::optional<T>& value) const -> std::optional<object_t> {
        if (value.has_value()) {
            const auto& tensor = value.value();
            return get_first_object_of_type<object_t>(tensor);
        }
        return std::nullopt;
    }
};

template <typename T>
struct get_first_object_of_type_t<std::vector<T>> {
    template <typename object_t>
    auto operator()(const std::vector<T>& value) const -> std::optional<object_t> {
        for (const auto& tensor : value) {
            auto result = get_first_object_of_type<object_t>(tensor);
            if (result.has_value()) {
                return result;
            }
        }
        return std::nullopt;
    }
};

template <typename T, auto N>
struct get_first_object_of_type_t<std::array<T, N>> {
    template <typename object_t>
    auto operator()(const std::array<T, N>& value) const -> std::optional<object_t> {
        for (const auto& tensor : value) {
            auto result = get_first_object_of_type<object_t>(tensor);
            if (result.has_value()) {
                return result;
            }
        }
        return std::nullopt;
    }
};

template <typename... Ts>
struct get_first_object_of_type_t<std::tuple<Ts...>> {
    template <typename object_t>
    auto operator()(const std::tuple<Ts...>& value) const -> std::optional<object_t> {
        if constexpr (sizeof...(Ts) > 0) {
            return get_first_object_of_type<object_t>(std::get<0>(value));
        } else {
            return std::nullopt;
        }
    }
};

template <typename T>
    requires requires { std::decay_t<T>::attribute_names; }
struct get_first_object_of_type_t<T> {
    template <typename object_t>
        requires(std::same_as<std::decay_t<T>, object_t>)
    auto operator()(const T& object) const -> std::optional<object_t> {
        return object;
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    auto operator()(const T& object) const -> std::optional<object_t> {
        return get_first_object_of_type<object_t>(object.attribute_values());
    }
};

namespace detail {
template <typename T>
concept has_first_member = requires(const T& obj) { reflect::get<0>(obj); };
}  // namespace detail

template <typename T>
    requires ttsl::concepts::Reflectable<std::decay_t<T>> && (not requires { std::decay_t<T>::attribute_names; })
struct get_first_object_of_type_t<T> {
    template <typename object_t>
        requires(std::same_as<std::decay_t<T>, object_t>)
    auto operator()(const T& object) const -> std::optional<object_t> {
        return object;
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>) && detail::has_first_member<T>
    auto operator()(const T& object) const -> std::optional<object_t> {
        std::optional<object_t> result = std::nullopt;
        reflect::for_each(
            [&result, &object](auto I) {
                if (!result.has_value()) {
                    result = get_first_object_of_type<object_t>(reflect::get<I>(object));
                }
            },
            object);
        return result;
    }

    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>) && (!detail::has_first_member<T>)
    auto operator()(const T& /*object*/) const -> std::optional<object_t> {
        return std::nullopt;
    }
};

}  // namespace ttsl::reflection
