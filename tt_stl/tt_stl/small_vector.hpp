// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/llvm/llvm_small_vector.hpp>

#include <tt_stl/reflection.hpp>

namespace ttsl {

static constexpr size_t SMALL_VECTOR_SIZE = 8;

template <typename T, std::size_t PREALLOCATED_SIZE = SMALL_VECTOR_SIZE>
struct SmallVector : public ttsl::detail::llvm::SmallVector<T, PREALLOCATED_SIZE> {
    using ttsl::detail::llvm::SmallVector<T, PREALLOCATED_SIZE>::SmallVector;
};

template <typename Stream, typename T, std::size_t PREALLOCATED_SIZE>
Stream& operator<<(Stream& os, const SmallVector<T, PREALLOCATED_SIZE>& vec) {
    os << "SmallVector([";
    for (auto i = 0; i < vec.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        using ttsl::reflection::operator<<;
        os << vec[i];
    }
    os << "])";
    return os;
}

}  // namespace ttsl

namespace ttnn {
template <typename T, size_t PREALLOCATED_SIZE = ttsl::SMALL_VECTOR_SIZE>
using SmallVector [[deprecated("Use ttsl::SmallVector instead")]] = tt::stl::SmallVector<T, PREALLOCATED_SIZE>;
}

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt

template <typename T, size_t PREALLOCATED_SIZE>
struct std::hash<ttsl::SmallVector<T, PREALLOCATED_SIZE>> {
    size_t operator()(const ttsl::SmallVector<T, PREALLOCATED_SIZE>& vec) const noexcept {
        size_t hash = 0;
        for (const auto& element : vec) {
            hash = ttsl::hash::detail::hash_objects(hash, element);
        }
        return hash;
    }
};

template <typename T, size_t PREALLOCATED_SIZE>
struct fmt::formatter<ttsl::SmallVector<T, PREALLOCATED_SIZE>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const ttsl::SmallVector<T, PREALLOCATED_SIZE>& vector, format_context& ctx) const
        -> format_context::iterator {
        std::stringstream ss;
        ss << vector;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T, size_t PREALLOCATED_SIZE>
struct ttsl::json::to_json_t<ttsl::SmallVector<T, PREALLOCATED_SIZE>> {
    nlohmann::json operator()(const ttsl::SmallVector<T, PREALLOCATED_SIZE>& vector) const {
        nlohmann::json json_array = nlohmann::json::array();
        for (const auto& element : vector) {
            json_array.push_back(to_json(element));
        }
        return json_array;
    }
};

template <typename T, size_t PREALLOCATED_SIZE>
struct ttsl::json::from_json_t<ttsl::SmallVector<T, PREALLOCATED_SIZE>> {
    ttsl::SmallVector<T, PREALLOCATED_SIZE> operator()(const nlohmann::json& json_object) const {
        ttsl::SmallVector<T, PREALLOCATED_SIZE> vector;
        for (const auto& element : json_object) {
            vector.push_back(from_json<T>(element));
        }
        return vector;
    }
};
