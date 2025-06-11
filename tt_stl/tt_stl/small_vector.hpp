// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <boost/container/small_vector.hpp>

#include <tt_stl/reflection.hpp>

namespace tt::stl {

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

}  // namespace tt::stl

namespace ttnn {
template <typename T, size_t PREALLOCATED_SIZE = tt::stl::SMALL_VECTOR_SIZE>
using SmallVector [[deprecated("Use tt::stl::SmallVector instead")]] = tt::stl::SmallVector<T, PREALLOCATED_SIZE>;
}

template <typename T, size_t PREALLOCATED_SIZE>
struct std::hash<tt::stl::SmallVector<T, PREALLOCATED_SIZE>> {
    size_t operator()(const tt::stl::SmallVector<T, PREALLOCATED_SIZE>& vec) const noexcept {
        size_t hash = 0;
        for (const auto& element : vec) {
            hash = tt::stl::hash::detail::hash_objects(hash, element);
        }
        return hash;
    }
};

template <typename T, size_t PREALLOCATED_SIZE>
struct fmt::formatter<tt::stl::SmallVector<T, PREALLOCATED_SIZE>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::stl::SmallVector<T, PREALLOCATED_SIZE>& vector, format_context& ctx) const
        -> format_context::iterator {
        std::stringstream ss;
        ss << vector;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T, size_t PREALLOCATED_SIZE>
struct tt::stl::json::to_json_t<tt::stl::SmallVector<T, PREALLOCATED_SIZE>> {
    nlohmann::json operator()(const tt::stl::SmallVector<T, PREALLOCATED_SIZE>& vector) const {
        nlohmann::json json_array = nlohmann::json::array();
        for (const auto& element : vector) {
            json_array.push_back(to_json(element));
        }
        return json_array;
    }
};

template <typename T, size_t PREALLOCATED_SIZE>
struct tt::stl::json::from_json_t<tt::stl::SmallVector<T, PREALLOCATED_SIZE>> {
    tt::stl::SmallVector<T, PREALLOCATED_SIZE> operator()(const nlohmann::json& json_object) const {
        tt::stl::SmallVector<T, PREALLOCATED_SIZE> vector;
        for (const auto& element : json_object) {
            vector.push_back(from_json<T>(element));
        }
        return vector;
    }
};
