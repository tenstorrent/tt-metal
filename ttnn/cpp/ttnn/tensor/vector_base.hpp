// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <array>

#include <boost/container/small_vector.hpp>
#include <pybind11/stl.h>

#include "tt_metal/tt_stl/reflection.hpp"

namespace ttnn {

static constexpr size_t SMALL_VECTOR_SIZE = 8;

template <typename T>
struct SmallVector: public boost::container::small_vector<T, SMALL_VECTOR_SIZE> {
    using boost::container::small_vector<T, SMALL_VECTOR_SIZE>::small_vector;
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const SmallVector<T> &vec) {
    os << "SmallVector([";
    for (auto i = 0; i < vec.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << vec[i];
    }
    os << "])";
    return os;
}

// Container wrapper that allows negative indexing
class VectorBase final {
public:
    using Container = SmallVector<uint32_t>;

    VectorBase() = default;
    explicit VectorBase(const Container& shape) : m_value(shape) { init(); }
    explicit VectorBase(Container&& shape) : m_value(std::move(shape)) { init(); }
    explicit VectorBase(std::initializer_list<uint32_t> ilist) : m_value(ilist) { init(); }
    template<size_t N>
    explicit VectorBase(const std::array<uint32_t, N>& arr) : m_value(arr.begin(), arr.end()) { init(); }

    size_t size() const;

    template<size_t N>
    bool operator==(const std::array<uint32_t, N> &other) const {
        return m_value.size() == N && std::equal(m_value.begin(), m_value.end(), other.begin());
    }

    bool operator==(const VectorBase &other) const;
    bool operator==(const Container &other) const;

    uint32_t operator[](int32_t index) const;
    uint32_t &operator[](int32_t index);

    Container::const_iterator cbegin() const;
    Container::const_iterator cend() const;

    std::span<const uint32_t> view() const;

    static constexpr auto attribute_names = std::forward_as_tuple("value", "original_size");
    const auto attribute_values() const { return std::forward_as_tuple(this->m_value, this->m_original_size); }

private:
    void init();

    Container m_value;
    size_t m_original_size = 0;
};

}

template<typename T>
struct std::hash<ttnn::SmallVector<T>> {
    size_t operator()(const ttnn::SmallVector<T>& vec) const noexcept {
        size_t hash = 0;
        for (const auto& element : vec) {
            hash = tt::stl::hash::detail::hash_objects(hash, element);
        }
        return hash;
    }
};

template <typename T>
struct fmt::formatter<ttnn::SmallVector<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const ttnn::SmallVector<T>& vector, format_context& ctx) const -> format_context::iterator {
        std::stringstream ss;
        ss << vector;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

namespace PYBIND11_NAMESPACE { namespace detail {
    template <typename T>
    struct type_caster<ttnn::SmallVector<T>> : list_caster<ttnn::SmallVector<T>, T> {};
}}
