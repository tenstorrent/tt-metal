// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shape_base.hpp"
#include <stdexcept>
#include "fmt/color.h"
#include "tt_metal/common/assert.hpp"

namespace ttnn {

namespace {

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
        TT_THROW("ShapeBase[] index out of range. {} not in [{}, {})", index, -full_size, full_size);
    }

    return fixed_index;
}
}

void ShapeBase::init() {
    m_original_size = m_value.size();
    const size_t min_internal_size = 4;

    if(m_original_size < min_internal_size) {
        Container ones(min_internal_size - m_original_size, 1);
        m_value.insert(m_value.begin(), ones.begin(), ones.end());
    }
}

size_t ShapeBase::size() const {
    return m_original_size;
}

std::span<const uint32_t> ShapeBase::view() const {
    return std::span<const uint32_t>(cbegin(), cend());
}

bool ShapeBase::operator==(const ShapeBase &other) const = default;

bool ShapeBase::operator==(const Container &other) const {
    auto original_view = view();
    return std::equal(original_view.begin(), original_view.end(), other.begin(), other.end());
}

bool ShapeBase::operator==(const std::vector<uint32_t> &other) const {
    auto original_view = view();
    return std::equal(original_view.begin(), original_view.end(), other.begin(), other.end());
}

uint32_t ShapeBase::operator[](int32_t index) const {
    auto norm_index = normalized_index(index, m_original_size, m_value.size());
    return m_value[norm_index];
}

uint32_t& ShapeBase::operator[](int32_t index) {
    auto norm_index = normalized_index(index, m_original_size, m_value.size());
    return m_value[norm_index];
}

ShapeBase::Container::const_iterator ShapeBase::cbegin() const {
    return this->m_value.cbegin() + (m_value.size() - m_original_size);
}

ShapeBase::Container::const_iterator ShapeBase::cend() const {
    return this->m_value.cend();
}

} // namespace ttnn
