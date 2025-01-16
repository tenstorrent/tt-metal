// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shape_base.hpp"
#include <stdexcept>
#include "fmt/color.h"
#include <tt-metalium/assert.hpp>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

int32_t normalized_index(int32_t index, size_t original_size, size_t container_size) {
    int32_t orig_size = static_cast<int32_t>(original_size);
    int32_t full_size = static_cast<int32_t>(container_size);

    int fixed_index = index;
    if (fixed_index < 0) {
        fixed_index += full_size;
    } else {
        fixed_index += full_size - orig_size;
    }

    if (fixed_index < 0 || fixed_index >= full_size) {
        TT_THROW("ShapeBase[] index out of range. {} not in [{}, {})", index, -full_size, orig_size);
    }

    return fixed_index;
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

void ShapeBase::init() {
    original_size_ = value_.size();
    const size_t min_internal_size = 4;

    if (original_size_ < min_internal_size) {
        Container ones(min_internal_size - original_size_, 1);
        value_.insert(value_.begin(), ones.begin(), ones.end());
    }
}

bool ShapeBase::empty() const { return original_size_ == 0; }

size_t ShapeBase::size() const { return original_size_; }

std::span<const uint32_t> ShapeBase::view() const { return std::span<const uint32_t>(cbegin(), cend()); }

bool ShapeBase::operator==(const ShapeBase& other) const = default;

bool ShapeBase::operator==(const Container& other) const {
    auto original_view = view();
    return std::equal(original_view.begin(), original_view.end(), other.begin(), other.end());
}

bool ShapeBase::operator==(const std::vector<uint32_t>& other) const {
    auto original_view = view();
    return std::equal(original_view.begin(), original_view.end(), other.begin(), other.end());
}

uint32_t ShapeBase::operator[](int32_t index) const {
    auto norm_index = CMAKE_UNIQUE_NAMESPACE::normalized_index(index, original_size_, value_.size());
    return value_[norm_index];
}

uint32_t& ShapeBase::operator[](int32_t index) {
    auto norm_index = CMAKE_UNIQUE_NAMESPACE::normalized_index(index, original_size_, value_.size());
    return value_[norm_index];
}

ShapeBase::Container::const_iterator ShapeBase::cbegin() const {
    return this->value_.cbegin() + (value_.size() - original_size_);
}

ShapeBase::Container::const_iterator ShapeBase::cend() const { return this->value_.cend(); }

}  // namespace tt::tt_metal
