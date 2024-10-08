// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/core.h>

#include <experimental/type_traits>
#include <optional>
#include <ostream>
#include <reflect>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
#include <filesystem>

#include "concepts.hpp"
#include "third_party/json/json.hpp"
#include <magic_enum.hpp>
#include "type_name.hpp"
#include "tt_metal/common/logger.hpp"

namespace tt {
namespace stl {

template <typename T>
constexpr std::string_view get_type_name() {
    return short_type_name<std::decay_t<T>>;
}

template <typename T>
constexpr std::string_view get_type_name(const T& object) {
    return get_type_name<T>();
}

template <typename T>
concept IsVariant = requires { typename std::variant_size<T>::type; };

template <IsVariant Variant>
constexpr auto get_active_type_name_in_variant(const Variant& v) {
    return std::visit([](auto&& arg) -> std::string_view { return short_type_name<std::decay_t<decltype(arg)>>; }, v);
}

namespace detail {
template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};
}  // namespace detail

template <typename Test, template <typename...> class Ref>
constexpr bool is_specialization_v = detail::is_specialization<Test, Ref>::value;

// Forward Declare hash_object
namespace hash {

constexpr bool DEBUG_HASH_OBJECT_FUNCTION = false;

using hash_t = std::uint64_t;
constexpr hash_t DEFAULT_SEED = 1234;

// stuff this in a header somewhere
inline int type_hash_counter = 0;
template <typename T>
inline const int type_hash = type_hash_counter++;

namespace detail {

template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>&) noexcept;

template <typename... Ts>
inline hash_t hash_object(const std::variant<Ts...>&) noexcept;

template <typename... Ts>
inline hash_t hash_object(const std::reference_wrapper<Ts...>&) noexcept;

template <typename T>
inline hash_t hash_object(const T&) noexcept;

template <typename... Types>
inline hash_t hash_objects(hash_t, const Types&...) noexcept;

}  // namespace detail

}  // namespace hash

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

}  // namespace json

namespace reflection {

template<std::size_t Start, std::size_t End, class T, std::size_t Size>
consteval auto fixed_string_substring(reflect::fixed_string<T, Size> string) {
    constexpr auto new_size = End - Start;
    T new_data[new_size];
    for (auto index = 0; index < new_size; index++) {
        new_data[index] = string.data[index + Start];
    }
    return reflect::fixed_string<T, new_size>{new_data};
}

template<class T1, std::size_t Size1, class T2, std::size_t Size2>
consteval auto fixed_string_equals(reflect::fixed_string<T1, Size1> string1, reflect::fixed_string<T2, Size2> string2) {
    if constexpr (string1.size() != string2.size()) {
        return false;
    } else {
        for (auto index = 0; index < string1.size(); index++) {
            if (string1.data[index] != string2.data[index]) {
                return false;
            }
        }
        return true;
    }
}

using AttributeName = std::string;

struct Attribute final {
    static constexpr std::size_t ALIGNMENT = 32;
    using storage_t = std::array<std::byte, 1312>;

    const std::string to_string() const { return this->implementations.to_string_impl_(this->type_erased_storage); }
    const std::size_t to_hash() const { return this->implementations.to_hash_impl_(this->type_erased_storage); }
    const nlohmann::json to_json() const { return this->implementations.to_json_impl_(this->type_erased_storage); }

    template <typename Type, typename BaseType = std::decay_t<Type>>
    Attribute(Type&& object) :
        pointer{new(&type_erased_storage) BaseType{std::forward<Type>(object)}},
        delete_storage{[](storage_t& self) { reinterpret_cast<BaseType*>(&self)->~BaseType(); }},
        copy_storage{[](storage_t& self, const void* other) -> void* {
            if constexpr (std::is_copy_constructible_v<BaseType>) {
                return new (&self) BaseType{*reinterpret_cast<const BaseType*>(other)};
            } else {
                static_assert(tt::stl::concepts::always_false_v<BaseType>);
            }
        }},
        move_storage{[](storage_t& self, void* other) -> void* {
            if constexpr (std::is_move_constructible_v<BaseType>) {
                return new (&self) BaseType{*reinterpret_cast<BaseType*>(other)};
            } else {
                static_assert(tt::stl::concepts::always_false_v<BaseType>);
            }
        }},

        implementations{
            .to_string_impl_ = [](const storage_t& storage) -> const std::string {
                const auto& object = *reinterpret_cast<const BaseType*>(&storage);
                if constexpr (std::is_pointer_v<BaseType>) {
                    return fmt::format("{}*", get_type_name<BaseType>());
                } else {
                    return fmt::format("{}", object);
                }
            },
            .to_hash_impl_ = [](const storage_t& storage) -> const std::size_t {
                const auto& object = *reinterpret_cast<const BaseType*>(&storage);
                return hash::detail::hash_object(object);
            },
            .to_json_impl_ = [](const storage_t& storage) -> const nlohmann::json {
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

    Attribute& operator=(const Attribute& other) {
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

    Attribute(Attribute&& other) :
        pointer{other.pointer ? other.move_storage(this->type_erased_storage, other.pointer) : nullptr},
        delete_storage{other.delete_storage},
        copy_storage{other.copy_storage},
        move_storage{other.move_storage},
        implementations{other.implementations} {}

    Attribute& operator=(Attribute&& other) {
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
    alignas(ALIGNMENT) void* pointer = nullptr;
    alignas(ALIGNMENT) storage_t type_erased_storage;

    void (*delete_storage)(storage_t&) = nullptr;
    void* (*copy_storage)(storage_t& storage, const void*) = nullptr;
    void* (*move_storage)(storage_t& storage, void*) = nullptr;

    struct implementations_t {
        const std::string (*to_string_impl_)(const storage_t&) = nullptr;
        const std::size_t (*to_hash_impl_)(const storage_t&) = nullptr;
        const nlohmann::json (*to_json_impl_)(const storage_t&) = nullptr;
    };

    implementations_t implementations;
};

using Attributes = std::vector<std::tuple<AttributeName, Attribute>>;

namespace detail {
template <typename T>
using has_to_hash_t = decltype(std::declval<const T>().to_hash());

template <typename T>
constexpr bool supports_to_hash_v = std::experimental::is_detected_v<has_to_hash_t, T>;

template <typename T>
using has_to_string_t = decltype(std::declval<const T>().to_string());

template <typename T>
constexpr bool supports_to_string_v = std::experimental::is_detected_v<has_to_string_t, T>;

template <typename T>
static constexpr std::size_t get_num_attributes() {
    static_assert(
        std::tuple_size_v<decltype(T::attribute_names)> ==
            std::tuple_size_v<decltype(std::declval<T>().attribute_values())>,
        "Number of attribute_names must match number of attribute_values");
    return std::tuple_size_v<decltype(T::attribute_names)>;
}
template <typename T>
using has_attribute_names_t = decltype(std::declval<T>().attribute_names);

template <typename T>
using has_attribute_values_t = decltype(std::declval<T>().attribute_values());

template <typename T>
constexpr bool supports_compile_time_attributes_v = std::experimental::is_detected_v<has_attribute_names_t, T> and
                                                    std::experimental::is_detected_v<has_attribute_values_t, T>;

template <typename T>
constexpr bool supports_conversion_to_string_v =
    detail::supports_to_string_v<T> or detail::supports_compile_time_attributes_v<T>;
}  // namespace detail

template <typename T>
Attributes get_attributes(const T& object) {
    if constexpr (tt::stl::reflection::detail::supports_compile_time_attributes_v<std::decay_t<T>>) {
        constexpr auto num_attributes = tt::stl::reflection::detail::get_num_attributes<std::decay_t<T>>();
        tt::stl::reflection::Attributes attributes;
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
    } else if constexpr (tt::stl::concepts::Reflectable<std::decay_t<T>>) {
        tt::stl::reflection::Attributes attributes;
        reflect::for_each(
            [&object, &attributes](auto I) {
                const auto& attribute_name = reflect::member_name<I>(object);
                const auto& attribute = reflect::get<I>(object);
                attributes.push_back({std::string{attribute_name}, attribute});
            },
            object);
        return attributes;
    } else {
        return object.attributes();
    }
}

static std::ostream& operator<<(std::ostream& os, const Attribute& attribute) {
    os << attribute.to_string();
    return os;
}

template <typename T>
typename std::enable_if_t<detail::supports_conversion_to_string_v<T>, std::ostream>& operator<<(
    std::ostream& os, const T& object) {
    if constexpr (detail::supports_to_string_v<T>) {
        os << object.to_string();
    } else if constexpr (detail::supports_compile_time_attributes_v<T>) {
        constexpr auto num_attributes = detail::get_num_attributes<T>();
        os << get_type_name<T>();
        os << "(";

        if constexpr (num_attributes > 0) {
            const auto attribute_values = object.attribute_values();
            [&os, &object, &attribute_values]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                (
                    [&os, &object, &attribute_values] {
                        const auto& attribute = std::get<Ns>(attribute_values);
                        os << std::get<Ns>(object.attribute_names);
                        os << "=";
                        os << attribute;
                        os << ",";
                    }(),
                    ...);
            }(std::make_index_sequence<num_attributes - 1>{});

            const auto& attribute = std::get<num_attributes - 1>(attribute_values);
            os << std::get<num_attributes - 1>(object.attribute_names);
            os << "=";
            os << attribute;
        }

        os << ")";
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Type cannot be converted to string");
    }
    return os;
}

template <typename T>
typename std::enable_if_t<std::is_enum<T>::value, std::ostream>& operator<<(std::ostream& os, const T& value) {
    os << magic_enum::enum_type_name<T>() << "::" << magic_enum::enum_name(value);
    return os;
}

static std::ostream& operator<<(std::ostream& os, const std::filesystem::path& path) {
    os << path.c_str();
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& optional) {
    if (optional.has_value()) {
        os << optional.value();
    } else {
        os << "std::nullopt";
    }
    return os;
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::variant<Ts...>& variant) {
    std::visit([&os](const auto& value) { os << value; }, variant);
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::reference_wrapper<T> reference_wrapper) {
    os << reference_wrapper.get();
    return os;
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& tuple) {
    [&os, &tuple]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&os, &tuple]() { os << (Ns == 0 ? "" : ", ") << std::get<Ns>(tuple); }(), ...);
    }(std::make_index_sequence<sizeof...(Ts)>{});
    return os;
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& array) {
    os << "{";
    for (auto index = 0; index < array.size(); index++) {
        const auto& element = array[index];
        os << element;
        if (index != array.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
    os << "{";
    for (auto index = 0; index < vector.size(); index++) {
        const T& element = vector[index];
        os << element;
        if (index != vector.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& set) {
    os << "{";
    auto index = 0;
    for (const auto& element : set) {
        os << element;
        if (index != set.size() - 1) {
            os << ", ";
        }
        index++;
    }
    os << "}";
    return os;
}

template <typename T>
    requires(tt::stl::concepts::Reflectable<T> and not(std::integral<T> or std::is_array<T>::value))
std::ostream& operator<<(std::ostream& os, const T& object) {
    os << reflect::type_name(object);
    os << "(";

    reflect::for_each(
        [&os, &object](auto I) {
            os << reflect::member_name<I>(object) << "=" << reflect::get<I>(object);
            if (I < reflect::size(object) - 1) {
                os << ",";
            }
        },
        object);

    os << ")";
    return os;
}

template <typename T>
struct visit_object_of_type_t;

template <typename object_t, typename T>
void visit_object_of_type(auto&& callback, T&& object) { visit_object_of_type_t<std::decay_t<T>>{}.template operator()<object_t>(callback, object); }

template<typename T>
requires (not tt::stl::concepts::Reflectable<std::decay_t<T>>) and (not  requires { std::decay_t<T>::attribute_names; })
struct visit_object_of_type_t<T> {

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, T&& value) const {
        callback(value);
    }

    template<typename object_t>
    requires (not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, T&& value) const {
        TT_THROW("Unsupported visit of object of type: {}", get_type_name<T>());
    }

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, const T& value) const {
        callback(value);
    }

    template<typename object_t>
    requires (not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, const T& value) const {
        TT_THROW("Unsupported visit of object of type: {}", get_type_name<T>());
    }
};

template<typename T>
struct visit_object_of_type_t<std::optional<T>> {

    template<typename object_t>
    void operator()(auto&& callback, const std::optional<T>& value) const {
        if (value.has_value()) {
            visit_object_of_type<object_t>(callback, value.value());
        }
    }
};

template<typename T>
struct visit_object_of_type_t<std::vector<T>> {

    template<typename object_t>
    void operator()(auto&& callback, const std::vector<T>& value) const {
        for (auto& tensor : value) {
            visit_object_of_type<object_t>(callback, tensor);
        }
    }
};

template<typename T, auto N>
struct visit_object_of_type_t<std::array<T, N>> {

    template<typename object_t>
    void operator()(auto&& callback, const std::array<T, N>& value) const {
        for (auto& tensor : value) {
            visit_object_of_type<object_t>(callback, tensor);
        }
    }
};

template<typename... Ts>
struct visit_object_of_type_t<std::tuple<Ts...>> {

    template<typename object_t>
    void operator()(auto&& callback, const std::tuple<Ts...>& value) const {
        [&callback, &value]<size_t... Ns>(std::index_sequence<Ns...>) {
            (visit_object_of_type<object_t>(callback, std::get<Ns>(value)), ...);
        }(std::make_index_sequence<sizeof...(Ts)>{});
    }
};

template<typename T>
requires requires { std::decay_t<T>::attribute_names; }
struct visit_object_of_type_t<T> {

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, T&& value) const {
        callback(value);
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, T&& object) const {
        constexpr auto num_attributes = std::tuple_size_v<decltype(std::decay_t<T>::attribute_names)>;
        visit_object_of_type<object_t>(callback, object.attribute_values());
    }

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, const T& value) const {
        callback(value);
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, const T& object) const {
        constexpr auto num_attributes = std::tuple_size_v<decltype(std::decay_t<T>::attribute_names)>;
        visit_object_of_type<object_t>(callback, object.attribute_values());
    }
};

template<typename T>
requires tt::stl::concepts::Reflectable<std::decay_t<T>>
struct visit_object_of_type_t<T> {

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, T&& value) const {
        callback(value);
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, T&& object) const {
        reflect::for_each(
            [&callback, &object](auto I) { visit_object_of_type<object_t>(callback, reflect::get<I>(object)); }, object);
    }

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    void operator()(auto&& callback, const T& value) const {
        callback(value);
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    void operator()(auto&& callback, const T& object) const {
        reflect::for_each(
            [&callback, &object](auto I) { visit_object_of_type<object_t>(callback, reflect::get<I>(object)); }, object);
    }
};

template <typename T>
struct transform_object_of_type_t;

template <typename object_t, typename T>
auto transform_object_of_type(auto&& callback, T&& object) { return transform_object_of_type_t<std::decay_t<T>>{}.template operator()<object_t>(callback, object); }

template<typename T>
requires (not tt::stl::concepts::Reflectable<std::decay_t<T>>) and (not  requires { std::decay_t<T>::attribute_names; })
struct transform_object_of_type_t<T> {

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, T&& value) const {
        return callback(value);
    }

    template<typename object_t>
    requires (not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& callback, T&& value) const {
        log_debug("Unsupported transform of object of type: {}. Do nothing.", get_type_name<T>());
        return value;
    }

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, const T& value) const {
        return callback(value);
    }

    template<typename object_t>
    requires (not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& callback, const T& value) const {
        log_debug("Unsupported transform of object of type: {}. Do nothing.", get_type_name<T>());
        return value;
    }
};

template<typename T>
struct transform_object_of_type_t<std::optional<T>> {

    template<typename object_t>
    std::optional<T> operator()(auto&& callback, const std::optional<T>& value) const {
        if (value.has_value()) {
            return transform_object_of_type<object_t>(callback, value.value());
        }
        return std::nullopt;
    }
};

template<typename T>
struct transform_object_of_type_t<std::vector<T>> {

    template<typename object_t>
    std::vector<T> operator()(auto&& callback, const std::vector<T>& value) const {
        std::vector<T> return_value;
        for (auto& tensor : value) {
            return_value.emplace_back(transform_object_of_type<object_t>(callback, tensor));
        }
        return return_value;
    }
};

template<typename T, auto N>
struct transform_object_of_type_t<std::array<T, N>> {

    template<typename object_t>
    std::array<T, N> operator()(auto&& callback, const std::array<T, N>& value) const {
        std::array<T, N> return_value;
        for (auto index = 0; index < value.size(); index++) {
            return_value[index] = transform_object_of_type<object_t>(callback, value[index]);
        }
        return return_value;
    }
};

template<typename... Ts>
struct transform_object_of_type_t<std::tuple<Ts...>> {

    template<typename object_t>
    std::tuple<Ts...> operator()(auto&& callback, const std::tuple<Ts...>& value) const {
        return std::apply(
            [&callback]<typename... args_t>(args_t&&... args) {
                return std::make_tuple(transform_object_of_type<object_t>(callback, std::forward<args_t>(args))...);
            },
            value);
    }
};

template<typename T>
requires requires { std::decay_t<T>::attribute_names; }
struct transform_object_of_type_t<T> {

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, T&& value) const {
        return callback(value);
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& callback, T&& object) const {
        static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported transform of object of type");
    }

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, const T& value) const {
        return callback(value);
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& callback, const T& object) const {
        static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported transform of object of type");
    }
};

template<typename T>
requires tt::stl::concepts::Reflectable<std::decay_t<T>>
struct transform_object_of_type_t<T> {

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, T&& value) const {
        return callback(value);
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& callback, T&& object) const {
        return std::apply(
            [&callback](auto&&... args) {
                return T{transform_object_of_type<object_t>(callback, std::forward<decltype(args)>(args))...};
            },
            reflect::to<std::tuple>(object));
    }

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    T operator()(auto&& callback, const T& value) const {
        return callback(value);
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    T operator()(auto&& callback, const T& object) const {
        return std::apply(
            [&callback](auto&&... args) {
                return T{transform_object_of_type<object_t>(callback, std::forward<decltype(args)>(args))...};
            },
            reflect::to<std::tuple>(object));
    }
};

template<typename T>
struct get_first_object_of_type_t;

template<typename object_t, typename T>
auto get_first_object_of_type(const T& value) {
    return get_first_object_of_type_t<std::decay_t<T>>{}.template operator()<object_t>(value);
}

template<typename T>
requires (not tt::stl::concepts::Reflectable<std::decay_t<T>>) and (not  requires { std::decay_t<T>::attribute_names; })
struct get_first_object_of_type_t<T> {

    template<typename object_t>
    requires std::same_as<std::decay_t<T>, object_t>
    const auto operator()(const T& value) const {
        return value;
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    const auto operator()(const T& value) const {
        TT_THROW("Unsupported get first object of type: {}", get_type_name<T>());
    }
};

template<typename T>
struct get_first_object_of_type_t<std::optional<T>> {

    template<typename object_t>
    const auto operator()(const std::optional<T>& value) const {
        if (value.has_value()) {
            const auto& tensor = value.value();
            return get_first_object_of_type<object_t>(tensor);
        }
    }
};

template<typename T>
struct get_first_object_of_type_t<std::vector<T>> {

    template<typename object_t>
    const auto operator()(const std::vector<T>& value) const {
        for (auto& tensor : value) {
            return get_first_object_of_type<object_t>(tensor);
        }
    }
};

template<typename T, auto N>
struct get_first_object_of_type_t<std::array<T, N>> {

    template<typename object_t>
    const auto operator()(const std::array<T, N>& value) const {
        for (auto& tensor : value) {
            return get_first_object_of_type<object_t>(tensor);
        }
    }
};

template<typename... Ts>
struct get_first_object_of_type_t<std::tuple<Ts...>> {

    template<typename object_t>
    const auto operator()(const std::tuple<Ts...>& value) const {
        return get_first_object_of_type<object_t>(std::get<0>(value));
    }
};

template<typename T>
requires requires { std::decay_t<T>::attribute_names; }
struct get_first_object_of_type_t<T> {

    template<typename object_t>
    requires(std::same_as<std::decay_t<T>, object_t>)
    const auto operator()(const T& object) const {
        return object;
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    const auto operator()(const T& object) const {
        constexpr auto num_attributes = std::tuple_size_v<decltype(std::decay_t<T>::attribute_names)>;
        return get_first_object_of_type<object_t>(object.attribute_values());
    }
};

template<typename T>
requires tt::stl::concepts::Reflectable<std::decay_t<T>>
struct get_first_object_of_type_t<T> {

    template<typename object_t>
    requires(std::same_as<std::decay_t<T>, object_t>)
    const auto operator()(const T& object) const {
        return object;
    }

    template<typename object_t>
    requires(not std::same_as<std::decay_t<T>, object_t>)
    const auto operator()(const T& object) const {
        return get_first_object_of_type<object_t>(reflect::get<0>(object));
    }
};

}  // namespace reflection
}  // namespace stl
}  // namespace tt

template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<tt::stl::reflection::detail::supports_conversion_to_string_v<T>>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const T& object, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << object;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<std::is_enum<T>::value>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const T& value, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << value;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <>
struct fmt::formatter<std::filesystem::path> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::filesystem::path& path, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << path;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::optional<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::optional<T>& optional, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << optional;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename... Ts>
struct fmt::formatter<std::variant<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::variant<Ts...>& variant, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << variant;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::reference_wrapper<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::reference_wrapper<T> reference, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << reference;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename... Ts>
struct fmt::formatter<std::tuple<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::tuple<Ts...>& tuple, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << tuple;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T, std::size_t N>
struct fmt::formatter<std::array<T, N>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::array<T, N>& array, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << array;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::vector<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::vector<T>& vector, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << vector;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::set<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::set<T>& set, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << set;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
    requires(
        tt::stl::concepts::Reflectable<T> and not(std::integral<T> or std::is_array<T>::value or
                                                  tt::stl::reflection::detail::supports_conversion_to_string_v<T>))
struct fmt::formatter<T> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const T& object, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << object;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

namespace tt {
namespace stl {
namespace hash {

namespace detail {
template <typename T, typename = std::void_t<>>
struct is_std_hashable : std::false_type {};

template <typename T>
struct is_std_hashable<T, std::void_t<decltype(std::declval<std::hash<T>>()(std::declval<T>()))>> : std::true_type {};

template <typename T>
constexpr bool is_std_hashable_v = is_std_hashable<T>::value;

template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>& array) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::array<{}, {}>: {}\n", get_type_name<T>(), N, array);
    }
    std::size_t hash = 0;
    [&array, &hash]<size_t... Ns>(std::index_sequence<Ns...>) {
        (
            [&array, &hash] {
                const auto& element = std::get<Ns>(array);
                hash = hash_objects(hash, element);
            }(),
            ...);
    }(std::make_index_sequence<N>{});
    return hash;
}

template <typename... Ts>
inline hash_t hash_object(const std::variant<Ts...>& variant) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::variant: {}\n", variant);
    }
    return std::visit([](const auto& value) { return hash_object(value); }, variant);
}

template <typename... Ts>
inline hash_t hash_object(const std::reference_wrapper<Ts...>& reference) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::reference_wrapper: {}\n", reference.get());
    }
    return hash_object(reference.get());
}

template <typename T>
inline hash_t hash_object(const T& object) noexcept {
    if constexpr (std::numeric_limits<T>::is_integer) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing integer of type {}: {}\n", get_type_name<T>(), object);
        }
        return object;
    } else if constexpr (detail::is_std_hashable_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing {} using std::hash: {}\n", get_type_name<T>(), object);
        }
        return std::hash<T>{}(object);
    } else if constexpr (tt::stl::reflection::detail::supports_to_hash_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using to_hash method: {}\n", get_type_name<T>(), object);
        }
        return object.to_hash();
    } else if constexpr (tt::stl::reflection::detail::supports_compile_time_attributes_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using compile-time attributes: {}\n", get_type_name<T>(), object);
        }
        constexpr auto num_attributes = reflection::detail::get_num_attributes<T>();
        std::size_t hash = 0;
        const auto attribute_values = object.attribute_values();
        [&object, &hash, &attribute_values]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&object, &hash, &attribute_values] {
                    const auto& attribute = std::get<Ns>(attribute_values);
                    hash = hash_objects(hash, attribute);
                }(),
                ...);
        }(std::make_index_sequence<num_attributes>{});
        return hash;
    } else if constexpr (is_specialization_v<T, std::vector>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::vector of type {}: {}\n", get_type_name<T>(), object);
        }
        auto hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (is_specialization_v<T, std::set>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::set of type {}: {}\n", get_type_name<T>(), object);
        }
        auto hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (is_specialization_v<T, std::optional>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::optional of type {}: {}\n", get_type_name<T>(), object);
        }
        if (object.has_value()) {
            return hash_object(object.value());
        } else {
            return 0;
        }
    } else if constexpr (tt::stl::concepts::Reflectable<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using reflect library: {}\n", get_type_name<T>(), object);
        }
        std::size_t hash = 0;
        reflect::for_each([&hash, &object](auto I) { hash = hash_objects(hash, reflect::get<I>(object)); }, object);
        return hash;
    } else {
        static_assert(
            tt::stl::concepts::always_false_v<T>, "Type doesn't support hashing using tt::stl::hash::hash_object");
    }
}

template <typename... Types>
inline hash_t hash_objects(hash_t seed, const Types&... args) noexcept {
    ([&seed](const auto& arg) { seed ^= hash_object(arg) + 0x9e3779b9 + (seed << 6) + (seed >> 2); }(args), ...);
    return seed;
}

}  // namespace detail

template <typename... Types>
inline hash_t hash_objects(hash_t seed, const Types&... args) noexcept {
    return detail::hash_objects(seed, args...);
}

template <typename... Types>
inline hash_t hash_objects_with_default_seed(const Types&... args) noexcept {
    return detail::hash_objects(DEFAULT_SEED, args...);
}

}  // namespace hash

namespace json {

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
    const char* operator()(const nlohmann::json& json_object) {
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

template <typename T>
    requires std::is_pointer_v<T>
struct to_json_t<T> {
    nlohmann::json operator()(const T& object) noexcept {
        if (object) {
            return to_json(*object);
        } else {
            return nullptr;
        }
    }
};

template <typename T>
    requires std::is_pointer_v<T>
struct from_json_t<T> {
    T operator()(const nlohmann::json& json_object) noexcept {
        if (json_object.is_null()) {
            return nullptr;
        } else {
            throw std::runtime_error("Cannot load pointer from JSON");
        }
    }
};

template <typename T, std::size_t N>
struct to_json_t<std::array<T, N>> {
    nlohmann::json operator()(const std::array<T, N>& array) noexcept {
        nlohmann::json json_array = nlohmann::json::array();
        std::size_t hash = 0;
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
        std::array<T, N> array;
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
    if constexpr (I >= std::variant_size_v<variant_t>)
        throw std::runtime_error{"Variant index " + std::to_string(I + index) + " out of bounds"};
    else
        return index == 0 ? from_json<std::variant_alternative_t<I, variant_t>>(json_object)
                          : variant_from_index<variant_t, I + 1>(index - 1, json_object);
}
}  // namespace detail

template <typename... Ts>
struct from_json_t<std::variant<Ts...>> {
    std::variant<Ts...> operator()(const nlohmann::json& json_object) {
        auto index = json_object["index"].get<std::size_t>();
        return detail::variant_from_index<std::variant<Ts...>>(index, json_object["value"]);
    }
};

template <typename T>
struct to_json_t<std::reference_wrapper<T>> {
    nlohmann::json operator()(const std::reference_wrapper<T>& reference) noexcept { return to_json(reference.get()); }
};

template <typename T>
struct to_json_t<std::optional<T>> {
    nlohmann::json operator()(const std::optional<T>& optional) noexcept {
        if (optional.has_value()) {
            return to_json(optional.value());
        } else {
            return nullptr;
        }
    }
};

template <typename T>
struct from_json_t<std::optional<T>> {
    std::optional<T> operator()(const nlohmann::json& json_object) noexcept {
        if (json_object.is_null()) {
            return std::nullopt;
        } else {
            return from_json<T>(json_object);
        }
    }
};

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

template <typename K, typename V>
struct to_json_t<std::map<K, V>> {
    nlohmann::json operator()(const std::map<K, V>& object) {
        nlohmann::json json_object = nlohmann::json::object();
        for (const auto& [key, value] : object) {
            json_object[to_json(key)] = to_json(value);
        }
        return json_object;
    }
};

template <typename K, typename V>
struct from_json_t<std::map<K, V>> {
    std::map<K, V> operator()(const nlohmann::json& json_object) {
        std::map<K, V> object;
        for (const auto& [key, value] : json_object.items()) {
            object[from_json<K>(key)] = from_json<V>(value);
        }
        return object;
    }
};

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

template <>
struct to_json_t<reflection::Attribute> {
    nlohmann::json operator()(const reflection::Attribute& attribute) noexcept { return attribute.to_json(); }
};

template <typename T>
    requires tt::stl::reflection::detail::supports_compile_time_attributes_v<T>
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
        }(std::make_index_sequence<reflection::detail::get_num_attributes<T>()>{});
        return json_object;
    }
};

template <typename T>
    requires tt::stl::concepts::Reflectable<T>
struct to_json_t<T> {
    nlohmann::json operator()(const T& object) noexcept {
        nlohmann::json json_object = nlohmann::json::object();
        reflect::for_each(
            [&object, &json_object](auto I) {
                const auto& attribute_name = reflect::member_name<I>(object);
                const auto& attribute = reflect::get<I>(object);
                json_object[std::string{attribute_name}] = to_json(attribute);
            },
            object);
        return json_object;
    }
};

template <typename T>
    requires tt::stl::concepts::Reflectable<T>
struct from_json_t<T> {
    T operator()(const nlohmann::json& json_object) noexcept {
        T object;
        reflect::for_each(
            [&object, &json_object](auto I) {
                const auto& attribute_name = reflect::member_name<I>(object);
                const auto& attribute = reflect::get<I>(object);
                reflect::get<I>(object) =
                    from_json<std::decay_t<decltype(attribute)>>(json_object[std::string{attribute_name}]);
            },
            object);
        return object;
    }
};

template <typename T>
struct to_json_t {
    nlohmann::json operator()(const T& optional) noexcept {
        return fmt::format("tt::stl::json::to_json_t: Unsupported type {}", get_type_name<T>());
    }
};

}  // namespace json

}  // namespace stl
}  // namespace tt
