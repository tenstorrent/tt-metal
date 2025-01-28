// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shape.hpp"

#include <numeric>
#include <ostream>
#include "ttnn/tensor/shape/small_vector.hpp"
#include <tt-metalium/assert.hpp>

namespace tt::tt_metal {

bool Shape::operator==(const Shape& other) const = default;

bool Shape::operator==(const SmallVector<uint32_t>& other) const { return this->value_ == other; }

size_t Shape::rank() const { return this->size(); }

uint64_t Shape::volume() const { return std::accumulate(cbegin(), cend(), uint64_t{1}, std::multiplies<uint64_t>()); }

std::array<uint32_t, 4> Shape::to_array_4D() const {
    TT_FATAL(rank() == 4, "to_array_4D is only valid for 4D shapes! Called for {}.", *this);
    std::array<uint32_t, 4> ret_array;
    for (int i = 0; i < rank(); i++) {
        ret_array[i] = this->operator[](i);
    }
    return ret_array;
}

Shape Shape::to_rank(size_t new_rank) const {
    SmallVector<uint32_t> new_shape(new_rank, 1);

    int cur_idx = static_cast<int>(rank()) - 1;
    int new_idx = static_cast<int>(new_rank) - 1;
    for (; cur_idx >= 0 && new_idx >= 0; cur_idx--, new_idx--) {
        new_shape[new_idx] = (*this)[cur_idx];
    }
    for (; cur_idx >= 0; cur_idx--) {
        TT_FATAL((*this)[cur_idx] == 1, "Can't convert shape rank");
    }

    return Shape(std::move(new_shape));
}

const uint32_t Shape::get_normalized_index(std::int64_t index) const {
    std::int64_t rank = static_cast<std::int64_t>(this->rank());
    std::uint64_t normalized_index = index >= 0 ? index : rank + index;
    TT_FATAL(
        normalized_index >= 0 and normalized_index < rank,
        "Index is out of bounds for the rank, should be between 0 and {} however is {}",
        rank - 1,
        normalized_index);
    return normalized_index;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Shape& shape) {
    os << "Shape([";
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "])";
    return os;
}

}  // namespace tt::tt_metal

#if TTNN_WITH_PYTHON_BINDINGS
namespace PYBIND11_NAMESPACE {
namespace detail {
namespace py = pybind11;
bool type_caster<ttnn::SimpleShape>::load(handle src, bool) {
    if (!py::isinstance<py::iterable>(src)) {
        return false;
    }
    ttnn::SmallVector<uint32_t> vec;
    for (auto item : src.cast<py::iterable>()) {
        if (!py::isinstance<py::int_>(item)) {
            return false;
        }
        vec.push_back(item.cast<uint32_t>());
    }
    value = ttnn::SimpleShape(std::move(vec));
    return true;
}

handle type_caster<ttnn::SimpleShape>::cast(
    const ttnn::SimpleShape& src, return_value_policy /* policy */, handle /* parent */) {
    py::list py_list;
    for (size_t i = 0; i < src.rank(); i++) {
        py_list.append(src[i]);
    }
    return py_list.release();
}
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
#endif
