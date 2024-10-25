// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "vector_base.hpp"
#include <stdexcept>
#include "fmt/color.h"
#include "tt_metal/common/assert.hpp"

namespace ttnn {

namespace {

constexpr size_t MIN_INTERNAL_SIZE = 4;

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
        TT_THROW("VectorBase[] index out of range. {} not in [{}, {})", index, -full_size, full_size);
    }

    return fixed_index;
}
}

void VectorBase::init() {
    m_original_size = m_value.size();

    if(m_original_size < MIN_INTERNAL_SIZE) {
        m_value.resize(MIN_INTERNAL_SIZE);
        size_t shift = MIN_INTERNAL_SIZE - m_original_size;
        for (size_t idx = MIN_INTERNAL_SIZE - 1; idx >= shift; idx--) {
            m_value[idx] = m_value[idx - shift];
        }
        for(size_t idx = 0; idx < shift; idx++) {
            m_value[idx] = 1;
        }
    }
}

size_t VectorBase::size() const {
    return m_original_size;
}

std::span<const uint32_t> VectorBase::view() const {
    return std::span<const uint32_t>(cbegin(), cend());
}

bool VectorBase::operator==(const VectorBase &other) const = default;

bool VectorBase::operator==(const Container &other) const {
    auto original_view = view();
    return std::equal(original_view.begin(), original_view.end(), other.cbegin(), other.cend());
}

uint32_t VectorBase::operator[](int32_t index) const {
    auto norm_index = normalized_index(index, m_original_size, m_value.size());
    return m_value[norm_index];
}

uint32_t& VectorBase::operator[](int32_t index) {
    auto norm_index = normalized_index(index, m_original_size, m_value.size());
    return m_value[norm_index];
}

VectorBase::Container::const_iterator VectorBase::cbegin() const {
    return this->m_value.cbegin() + (m_value.size() - m_original_size);
}

VectorBase::Container::const_iterator VectorBase::cend() const {
    return this->m_value.cend();
}

}
