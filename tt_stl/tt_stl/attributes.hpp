// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <fmt/base.h>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/concepts.hpp>
#include <tt_stl/hash.hpp>
#include <tt_stl/stl_fmt.hpp>
#include <tt_stl/stl_json.hpp>
#include <tt_stl/type_name.hpp>

// NOLINTBEGIN(bugprone-multi-level-implicit-pointer-conversion)

namespace ttsl {
namespace reflection {

using AttributeName = std::string;

struct Attribute final {
    static constexpr std::size_t ALIGNMENT = 32;
    using storage_t = std::array<std::byte, 1312>;

    std::string to_string() const { return this->implementations.to_string_impl_(this->type_erased_storage); }
    std::size_t to_hash() const { return this->implementations.to_hash_impl_(this->type_erased_storage); }
    nlohmann::json to_json() const { return this->implementations.to_json_impl_(this->type_erased_storage); }

    template <
        typename Type,
        typename BaseType = std::decay_t<Type>,
        std::enable_if_t<!std::is_same_v<BaseType, Attribute>, int> = 0>
    Attribute(Type&& object) :
        pointer{new(&type_erased_storage) BaseType{std::forward<Type>(object)}},
        delete_storage{[](storage_t& self) { reinterpret_cast<BaseType*>(&self)->~BaseType(); }},
        copy_storage{[](storage_t& self, const void* other) -> void* {
            if constexpr (std::is_copy_constructible_v<BaseType>) {
                return new (&self) BaseType{*reinterpret_cast<const BaseType*>(other)};
            } else {
                static_assert(ttsl::concepts::always_false_v<BaseType>);
            }
        }},
        move_storage{[](storage_t& self, void* other) -> void* {
            if constexpr (std::is_move_constructible_v<BaseType>) {
                return new (&self) BaseType{*reinterpret_cast<BaseType*>(other)};
            } else {
                static_assert(ttsl::concepts::always_false_v<BaseType>);
            }
        }},

        implementations{
            .to_string_impl_ = [](const storage_t& storage) -> std::string {
                const auto& object = *reinterpret_cast<const BaseType*>(&storage);
                if constexpr (std::is_pointer_v<BaseType>) {
                    return fmt::format("{}*", get_type_name<BaseType>());
                } else if constexpr (fmt::is_formattable<BaseType>::value) {
                    return fmt::format("{}", object);
                } else {
                    return std::string(get_type_name<BaseType>());
                }
            },
            .to_hash_impl_ = [](const storage_t& storage) -> std::size_t {
                const auto& object = *reinterpret_cast<const BaseType*>(&storage);
                if constexpr (hash::is_hashable_v<BaseType>) {
                    return hash::detail::hash_object(object);
                } else {
                    return static_cast<std::size_t>(hash::type_hash<BaseType>);
                }
            },
            .to_json_impl_ = [](const storage_t& storage) -> nlohmann::json {
                const auto& object = *reinterpret_cast<const BaseType*>(&storage);
                return json::to_json(object);
            }} {
        static_assert(sizeof(BaseType) <= sizeof(storage_t));
        static_assert(ALIGNMENT % alignof(BaseType) == 0);
    }

    void destruct() noexcept {
        if (this->pointer) {
            this->delete_storage(this->type_erased_storage);
        }
        this->pointer = nullptr;
    }

    Attribute(const Attribute& other) :
        pointer{other.pointer ? other.copy_storage(this->type_erased_storage, other.pointer) : nullptr},
        delete_storage{other.delete_storage},
        copy_storage{other.copy_storage},
        move_storage{other.move_storage},
        implementations{other.implementations} {}

    Attribute(Attribute&& other) noexcept :
        pointer{other.pointer ? other.move_storage(this->type_erased_storage, other.pointer) : nullptr},
        delete_storage{other.delete_storage},
        copy_storage{other.copy_storage},
        move_storage{other.move_storage},
        implementations{other.implementations} {}

    Attribute& operator=(const Attribute& other) {
        if (this == &other) {
            return *this;
        }
        if (other.pointer != this->pointer) {
            this->destruct();
            this->pointer = nullptr;
            if (other.pointer) {
                this->pointer = other.copy_storage(this->type_erased_storage, other.pointer);
            }
            this->delete_storage = other.delete_storage;
            this->copy_storage = other.copy_storage;
            this->move_storage = other.move_storage;
            this->implementations = other.implementations;
        }
        return *this;
    }

    Attribute& operator=(Attribute&& other) noexcept {
        if (other.pointer != this->pointer) {
            this->destruct();
            this->pointer = nullptr;
            if (other.pointer) {
                this->pointer = other.move_storage(this->type_erased_storage, other.pointer);
            }
            this->delete_storage = other.delete_storage;
            this->copy_storage = other.copy_storage;
            this->move_storage = other.move_storage;
            this->implementations = other.implementations;
        }
        return *this;
    }

    ~Attribute() { this->destruct(); }

private:
    alignas(ALIGNMENT) storage_t type_erased_storage{};
    void* pointer = nullptr;

    void (*delete_storage)(storage_t&) = nullptr;
    void* (*copy_storage)(storage_t& storage, const void*) = nullptr;
    void* (*move_storage)(storage_t& storage, void*) = nullptr;

    struct implementations_t {
        std::string (*to_string_impl_)(const storage_t&) = nullptr;
        std::size_t (*to_hash_impl_)(const storage_t&) = nullptr;
        nlohmann::json (*to_json_impl_)(const storage_t&) = nullptr;
    };

    implementations_t implementations;
};

using Attributes = std::vector<std::tuple<AttributeName, Attribute>>;

template <typename T>
Attributes get_attributes(const T& object) {
    if constexpr (ttsl::concepts::detail::supports_compile_time_attributes_v<std::decay_t<T>>) {
        constexpr auto num_attributes = ttsl::concepts::detail::get_num_attributes<std::decay_t<T>>();
        ttsl::reflection::Attributes attributes;
        const auto attribute_values = object.attribute_values();
        [&object, &attributes, &attribute_values]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&object, &attributes, &attribute_values] {
                    const auto& attribute_name = std::get<Ns>(object.attribute_names);
                    const auto& attribute = std::get<Ns>(attribute_values);
                    attributes.push_back({attribute_name, attribute});
                }(),
                ...);
        }(std::make_index_sequence<num_attributes>{});
        return attributes;
    } else if constexpr (requires { object.attributes(); }) {
        return object.attributes();
    } else {
        return {};
    }
}

inline std::ostream& operator<<(std::ostream& os, const Attribute& attribute) {
    os << attribute.to_string();
    return os;
}

// --- visit_object_of_type ---

template <typename T>
struct visit_object_of_type_t;

template <typename object_t, typename T>
void visit_object_of_type(auto&& callback, T&& object) {
    visit_object_of_type_t<std::decay_t<T>>{}.template operator()<object_t>(
        std::forward<decltype(callback)>(callback), std::forward<T>(object));
}

template <typename T>
    requires(not requires { std::decay_t<T>::attribute_names; })
struct visit_object_of_type_t<T> {
    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, T&& value) const {
        callback(value);
    }

    // Types without attribute_names cannot be traversed further.
    // This is a no-op: the object is silently skipped during visitation.
    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& /*callback*/, T&& /*value*/) const {}

    template <typename object_t>
        requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, const T& value) const {
        callback(value);
    }

    // Types without attribute_names cannot be traversed further.
    // This is a no-op: the object is silently skipped during visitation.
    template <typename object_t>
        requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& /*callback*/, const T& /*value*/) const {}
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
        visit_object_of_type<object_t>(callback, object.attribute_values());
    }
};

// --- transform_object_of_type ---

template <typename T>
struct transform_object_of_type_t;

template <typename object_t, typename T>
auto transform_object_of_type(auto&& callback, T&& object) {
    return transform_object_of_type_t<std::decay_t<T>>{}.template operator()<object_t>(
        std::forward<decltype(callback)>(callback), std::forward<T>(object));
}

template <typename T>
    requires(not requires { std::decay_t<T>::attribute_names; })
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

// --- get_first_object_of_type ---

template <typename T>
struct get_first_object_of_type_t;

template <typename object_t, typename T>
auto get_first_object_of_type(const T& value) {
    return get_first_object_of_type_t<std::decay_t<T>>{}.template operator()<object_t>(value);
}

template <typename T>
    requires(not requires { std::decay_t<T>::attribute_names; })
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
            std::optional<object_t> result;
            [&result, &value]<size_t... Ns>(std::index_sequence<Ns...>) {
                (
                    [&result, &value] {
                        if (!result.has_value()) {
                            result = get_first_object_of_type<object_t>(std::get<Ns>(value));
                        }
                    }(),
                    ...);
            }(std::make_index_sequence<sizeof...(Ts)>{});
            return result;
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

// --- JSON for Attribute ---

}  // namespace reflection

namespace json {

template <>
struct to_json_t<reflection::Attribute> {
    nlohmann::json operator()(const reflection::Attribute& attribute) noexcept { return attribute.to_json(); }
};

}  // namespace json
}  // namespace ttsl

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt

// NOLINTEND(bugprone-multi-level-implicit-pointer-conversion)
