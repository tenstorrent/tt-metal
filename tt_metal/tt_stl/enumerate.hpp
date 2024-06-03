// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

namespace tt::stl::utils {

template <
    typename Iterable,
    typename TIter = decltype(std::begin(std::declval<Iterable>())),
    typename = decltype(std::end(std::declval<Iterable>()))>
auto enumerate(Iterable&& iterable, std::size_t start = 0) {
    struct iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::tuple<std::size_t, typename std::iterator_traits<TIter>::value_type>;
        using difference_type = typename std::iterator_traits<TIter>::difference_type;
        using pointer = std::tuple<std::size_t, typename std::iterator_traits<TIter>::pointer>;
        using reference = std::tuple<std::size_t, typename std::iterator_traits<TIter>::reference>;

        bool operator!=(const iterator& other) const { return iter != other.iter; }
        void operator++() {
            ++index;
            ++iter;
        }
        auto operator*() const { return std::tie(index, *iter); }

        std::size_t index;
        TIter iter;
    };

    struct iterator_view {
        Iterable iterable;
        std::size_t start;

        auto begin() { return iterator{start, std::begin(iterable)}; }
        auto end() { return iterator{start, std::end(iterable)}; }
    };

    return iterator_view{std::forward<Iterable>(iterable), start};
}

}  // namespace tt::stl::utils
