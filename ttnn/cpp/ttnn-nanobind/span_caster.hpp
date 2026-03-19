// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/detail/nb_list.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include <tt_stl/span.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T, std::size_t Extent>
struct type_caster<ttsl::Span<T, Extent>> {
    using ValueItem = std::remove_cv_t<std::remove_reference_t<T>>;
    using ValueType = ttsl::Span<T, Extent>;
    using Storage = std::vector<ValueItem>;
    NB_TYPE_CASTER(
        ValueType,
        io_name("collections.abc.Sequence", "span") + const_name("[") + make_caster<ValueItem>::Name + const_name("]"));

    bool from_python(handle src, std::uint8_t, cleanup_list*) {
        if (!isinstance<list>(src)) {
            return false;
        }

        try {
            storage_ = nanobind::cast<Storage>(src);
        } catch (const nanobind::cast_error&) {
            return false;
        }

        if constexpr (Extent != std::dynamic_extent) {
            if (storage_.size() != Extent) {
                return false;
            }
        }

        this->value = ttsl::Span<T, Extent>(static_cast<T*>(storage_.data()), storage_.size());
        return true;
    }

    static handle from_cpp(const ttsl::Span<T, Extent>& src, rv_policy, cleanup_list*) {
        Storage storage(src.begin(), src.end());
        return nanobind::cast(std::move(storage)).release();
    }

private:
    Storage storage_;
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
