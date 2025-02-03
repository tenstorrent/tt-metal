// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <boost/container/small_vector.hpp>

#include <tt-metalium/reflection.hpp>

#if TTNN_WITH_PYTHON_BINDINGS
#include <pybind11/stl.h>
#endif

namespace tt::tt_metal {

static constexpr size_t SMALL_VECTOR_SIZE = 8;

template <typename T, size_t PREALLOCATED_SIZE = SMALL_VECTOR_SIZE>
struct SmallVector : public boost::container::small_vector<T, PREALLOCATED_SIZE> {
    using boost::container::small_vector<T, PREALLOCATED_SIZE>::small_vector;
};

template <typename T, size_t PREALLOCATED_SIZE>
std::ostream& operator<<(std::ostream& os, const SmallVector<T, PREALLOCATED_SIZE>& vec) {
    os << "SmallVector([";
    for (auto i = 0; i < vec.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        using tt::stl::reflection::operator<<;
        os << vec[i];
    }
    os << "])";
    return os;
}

}  // namespace tt::tt_metal

namespace ttnn {
using tt::tt_metal::SmallVector;
}

template <typename T, size_t PREALLOCATED_SIZE>
struct std::hash<tt::tt_metal::SmallVector<T, PREALLOCATED_SIZE>> {
    size_t operator()(const ttnn::SmallVector<T, PREALLOCATED_SIZE>& vec) const noexcept {
        size_t hash = 0;
        for (const auto& element : vec) {
            hash = tt::stl::hash::detail::hash_objects(hash, element);
        }
        return hash;
    }
};

template <typename T, size_t PREALLOCATED_SIZE>
struct fmt::formatter<tt::tt_metal::SmallVector<T, PREALLOCATED_SIZE>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::SmallVector<T, PREALLOCATED_SIZE>& vector, format_context& ctx) const
        -> format_context::iterator {
        std::stringstream ss;
        ss << vector;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

#if TTNN_WITH_PYTHON_BINDINGS
namespace PYBIND11_NAMESPACE {
namespace detail {
template <typename T, size_t PREALLOCATED_SIZE>
struct type_caster<tt::tt_metal::SmallVector<T, PREALLOCATED_SIZE>>
    : list_caster<tt::tt_metal::SmallVector<T, PREALLOCATED_SIZE>, T> {};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
#endif
