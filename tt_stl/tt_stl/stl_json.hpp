// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>
#include <tt_stl/concepts.hpp>
#include <tt_stl/type_name.hpp>

namespace ttsl {
namespace json {

template <typename T>
struct to_json_t;

nlohmann::json to_json(const auto& object) { return to_json_t<std::decay_t<decltype(object)>>{}(object); }

template <typename T>
struct from_json_t;

template <typename T>
T from_json(const nlohmann::json& json_object) {
    return from_json_t<T>{}(json_object);
}

// --- Primitive types ---

template <typename T>
    requires std::is_integral_v<T> or std::is_floating_point_v<T> or std::is_enum_v<T>
struct to_json_t<T> {
    nlohmann::json operator()(const T& object) noexcept { return T{object}; }
};

template <typename T>
    requires std::is_integral_v<T> or std::is_floating_point_v<T> or std::is_enum_v<T>
struct from_json_t<T> {
    T operator()(const nlohmann::json& json_object) noexcept { return json_object.get<T>(); }
};

template <>
struct to_json_t<const char*> {
    nlohmann::json operator()(const char* object) noexcept { return object; }
};

template <>
struct from_json_t<const char*> {
    const char* operator()(const nlohmann::json& /*json_object*/) {
        throw std::runtime_error("Cannot load const char* from JSON");
    }
};

template <>
struct to_json_t<std::string> {
    nlohmann::json operator()(const std::string& object) noexcept { return object; }
};

template <>
struct from_json_t<std::string> {
    std::string operator()(const nlohmann::json& json_object) noexcept { return json_object.get<std::string>(); }
};

// --- Pointer types ---

template <typename T>
    requires std::is_pointer_v<T>
struct to_json_t<T> {
    nlohmann::json operator()(const T& object) noexcept {
        if (object) {
            return to_json(*object);
        }
        return nullptr;
    }
};

template <typename T>
    requires std::is_pointer_v<T>
struct from_json_t<T> {
    T operator()(const nlohmann::json& json_object) noexcept {
        if (json_object.is_null()) {
            return nullptr;
        }
        throw std::runtime_error("Cannot load pointer from JSON");
    }
};

// --- std::array ---

template <typename T, std::size_t N>
struct to_json_t<std::array<T, N>> {
    nlohmann::json operator()(const std::array<T, N>& array) noexcept {
        nlohmann::json json_array = nlohmann::json::array();
        [&array, &json_array]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&array, &json_array] {
                    const auto& element = std::get<Ns>(array);
                    json_array.push_back(to_json(element));
                }(),
                ...);
        }(std::make_index_sequence<N>{});
        return json_array;
    }
};

template <typename T, std::size_t N>
struct from_json_t<std::array<T, N>> {
    std::array<T, N> operator()(const nlohmann::json& json_object) noexcept {
        std::array<T, N> array{};
        [&array, &json_object]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&array, &json_object] {
                    const auto& element = json_object[Ns];
                    array[Ns] = from_json<T>(element);
                }(),
                ...);
        }(std::make_index_sequence<N>{});
        return array;
    }
};

// --- std::variant ---

template <typename... Ts>
struct to_json_t<std::variant<Ts...>> {
    nlohmann::json operator()(const std::variant<Ts...>& variant) noexcept {
        return std::visit(
            [index = variant.index()](const auto& value) -> nlohmann::json {
                nlohmann::json json_object = nlohmann::json::object();
                return {{"index", index}, {"value", to_json(value)}};
            },
            variant);
    }
};

namespace detail {
template <class variant_t, std::size_t I = 0>
variant_t variant_from_index(std::size_t index, const nlohmann::json& json_object) {
    if constexpr (I >= std::variant_size_v<variant_t>) {
        throw std::runtime_error{"Variant index " + std::to_string(I + index) + " out of bounds"};
    } else {
        return index == 0 ? from_json<std::variant_alternative_t<I, variant_t>>(json_object)
                          : variant_from_index<variant_t, I + 1>(index - 1, json_object);
    }
}
}  // namespace detail

template <typename... Ts>
struct from_json_t<std::variant<Ts...>> {
    std::variant<Ts...> operator()(const nlohmann::json& json_object) {
        auto index = json_object["index"].get<std::size_t>();
        return detail::variant_from_index<std::variant<Ts...>>(index, json_object["value"]);
    }
};

// --- std::reference_wrapper ---

template <typename T>
struct to_json_t<std::reference_wrapper<T>> {
    nlohmann::json operator()(const std::reference_wrapper<T>& reference) noexcept { return to_json(reference.get()); }
};

// --- std::optional ---

template <typename T>
struct to_json_t<std::optional<T>> {
    nlohmann::json operator()(const std::optional<T>& optional) noexcept {
        if (optional.has_value()) {
            return to_json(optional.value());
        }
        return nullptr;
    }
};

template <typename T>
struct from_json_t<std::optional<T>> {
    std::optional<T> operator()(const nlohmann::json& json_object) noexcept {
        if (json_object.is_null()) {
            return std::nullopt;
        }
        return from_json<T>(json_object);
    }
};

// --- std::vector ---

template <typename T>
struct to_json_t<std::vector<T>> {
    nlohmann::json operator()(const std::vector<T>& vector) noexcept {
        nlohmann::json json_array = nlohmann::json::array();
        for (const auto& element : vector) {
            json_array.push_back(to_json(element));
        }
        return json_array;
    }
};

template <typename T>
struct from_json_t<std::vector<T>> {
    std::vector<T> operator()(const nlohmann::json& json_object) noexcept {
        std::vector<T> vector;
        for (const auto& element : json_object) {
            vector.push_back(from_json<T>(element));
        }
        return vector;
    }
};

// --- std::set ---

template <typename T>
struct to_json_t<std::set<T>> {
    nlohmann::json operator()(const std::set<T>& set) noexcept {
        nlohmann::json json_array = nlohmann::json::array();
        for (const auto& element : set) {
            json_array.push_back(to_json(element));
        }
        return json_array;
    }
};

template <typename T>
struct from_json_t<std::set<T>> {
    std::set<T> operator()(const nlohmann::json& json_object) noexcept {
        std::set<T> set;
        for (const auto& element : json_object) {
            set.insert(from_json<T>(element));
        }
        return set;
    }
};

// --- std::map ---

template <typename K, typename V>
struct to_json_t<std::map<K, V>> {
    nlohmann::json operator()(const std::map<K, V>& object) {
        nlohmann::json json_object = nlohmann::json::object();
        for (const auto& [key, value] : object) {
            json_object[to_json(key).dump()] = to_json(value);
        }
        return json_object;
    }
};

template <typename K, typename V>
struct from_json_t<std::map<K, V>> {
    std::map<K, V> operator()(const nlohmann::json& json_object) {
        std::map<K, V> object;
        for (const auto& [key, value] : json_object.items()) {
            object[from_json<K>(nlohmann::json::parse(key))] = from_json<V>(value);
        }
        return object;
    }
};

// --- std::unordered_map ---

template <typename K, typename V>
struct to_json_t<std::unordered_map<K, V>> {
    nlohmann::json operator()(const std::unordered_map<K, V>& object) {
        nlohmann::json json_object = nlohmann::json::object();
        for (const auto& [key, value] : object) {
            json_object[to_json(key).dump()] = to_json(value);
        }
        return json_object;
    }
};

template <typename K, typename V>
struct from_json_t<std::unordered_map<K, V>> {
    std::map<K, V> operator()(const nlohmann::json& json_object) {
        std::unordered_map<K, V> object;
        for (const auto& [key, value] : json_object.items()) {
            object[from_json<K>(nlohmann::json::parse(key))] = from_json<V>(value);
        }
        return object;
    }
};

// --- std::tuple ---

template <typename... Ts>
struct to_json_t<std::tuple<Ts...>> {
    nlohmann::json operator()(const std::tuple<Ts...>& tuple) noexcept {
        nlohmann::json json_array = nlohmann::json::array();
        [&tuple, &json_array]<std::size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&tuple, &json_array] {
                    const auto& element = std::get<Ns>(tuple);
                    json_array.push_back(to_json(element));
                }(),
                ...);
        }(std::make_index_sequence<sizeof...(Ts)>{});
        return json_array;
    }
};

template <typename... Ts>
struct from_json_t<std::tuple<Ts...>> {
    std::tuple<Ts...> operator()(const nlohmann::json& json_object) noexcept {
        std::tuple<Ts...> tuple;
        [&tuple, &json_object]<std::size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&tuple, &json_object] {
                    const auto& element = json_object[Ns];
                    std::get<Ns>(tuple) = from_json<Ts>(element);
                }(),
                ...);
        }(std::make_index_sequence<sizeof...(Ts)>{});
        return tuple;
    }
};

// --- Compile-time attributes types ---

template <typename T>
    requires ttsl::concepts::detail::supports_compile_time_attributes_v<T>
struct to_json_t<T> {
    nlohmann::json operator()(const T& object) noexcept {
        nlohmann::json json_object = nlohmann::json::object();
        const auto attribute_values = object.attribute_values();
        [&object, &json_object, &attribute_values]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&object, &json_object, &attribute_values] {
                    const auto& attribute_name = std::get<Ns>(object.attribute_names);
                    const auto& attribute = std::get<Ns>(attribute_values);
                    json_object[attribute_name] = to_json(attribute);
                }(),
                ...);
        }(std::make_index_sequence<concepts::detail::get_num_attributes<T>()>{});
        return json_object;
    }
};

template <typename T>
    requires ttsl::concepts::detail::supports_compile_time_attributes_v<T>
struct from_json_t<T> {
    T operator()(const nlohmann::json& json_object) {
        T obj{};
        // attribute_values() is const, but obj itself is mutable.
        // Use const_cast to get mutable references to the underlying fields.
        auto const_values = obj.attribute_values();
        [&obj, &json_object, &const_values]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&obj, &json_object, &const_values] {
                    const auto& attribute_name = std::get<Ns>(obj.attribute_names);
                    using field_type = std::decay_t<decltype(std::get<Ns>(const_values))>;
                    if (json_object.contains(attribute_name)) {
                        const_cast<field_type&>(std::get<Ns>(const_values)) =
                            from_json<field_type>(json_object.at(attribute_name));
                    }
                }(),
                ...);
        }(std::make_index_sequence<concepts::detail::get_num_attributes<T>()>{});
        return obj;
    }
};

// --- Fallback ---

template <typename T>
struct to_json_t {
    nlohmann::json operator()(const T& /*optional*/) noexcept {
        return std::string("ttsl::json::to_json_t: Unsupported type ") + std::string(get_type_name<T>());
    }
};

template <typename T>
struct from_json_t {
    T operator()(const nlohmann::json& /*json_object*/) {
        throw std::runtime_error(
            std::string("ttsl::json::from_json_t: Unsupported type ") + std::string(get_type_name<T>()));
    }
};

}  // namespace json
}  // namespace ttsl

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt
