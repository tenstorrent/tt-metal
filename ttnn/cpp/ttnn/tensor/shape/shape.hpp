// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "shape_base.hpp"

#if TTNN_WITH_PYTHON_BINDINGS
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

namespace tt::tt_metal {

class SimpleShape final : protected ShapeBase {
public:
    using ShapeBase::ShapeBase;
    using ShapeBase::operator[];
    using ShapeBase::cbegin;
    using ShapeBase::cend;
    using ShapeBase::empty;
    using ShapeBase::size;
    using ShapeBase::view;

    template <std::size_t N>
    bool operator==(const std::array<uint32_t, N>& other) const {
        const bool sameSize = value_.size() == N;
        return sameSize && std::equal(value_.begin(), value_.end(), other.begin());
    }

    bool operator==(const SimpleShape& other) const;
    bool operator==(const ShapeBase::Container& other) const;

    [[nodiscard]] size_t rank() const;
    [[nodiscard]] uint64_t volume() const;

    const uint32_t get_normalized_index(std::int64_t index) const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(this->value_); }

    std::array<uint32_t, 4> to_array_4D() const;

    friend std::ostream& operator<<(std::ostream& os, const SimpleShape& shape);
};

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::SimpleShape& shape);

}  // namespace tt::tt_metal

namespace ttnn {
using tt::tt_metal::SimpleShape;
}  // namespace ttnn

#if TTNN_WITH_PYTHON_BINDINGS
namespace PYBIND11_NAMESPACE {
namespace detail {
template <>
class type_caster<ttnn::SimpleShape> {
public:
    PYBIND11_TYPE_CASTER(ttnn::SimpleShape, _("SimpleShape"));

    bool load(handle src, bool);
    static handle cast(const ttnn::SimpleShape& src, return_value_policy /* policy */, handle /* parent */);
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
#endif
