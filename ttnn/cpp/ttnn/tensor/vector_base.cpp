// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "vector_base.hpp"
#include <stdexcept>
#include "fmt/color.h"
#include "tt_metal/common/assert.hpp"

namespace tt::tt_metal {

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
        TT_THROW("VectorBase[] index out of range. {} not in [{}, {})", index, -full_size, full_size);
    }

    return fixed_index;
}
}

void VectorBase::init() {
    m_original_size = m_value.size();
    const size_t min_internal_size = 4;
    if(m_original_size < min_internal_size) {
        std::vector<uint32_t> new_value(min_internal_size, 1);
        std::copy(m_value.begin(), m_value.end(), new_value.begin() + (min_internal_size - m_original_size));
        m_value = std::move(new_value);
    }
}

std::span<const uint32_t> VectorBase::view() const {
    return std::span<const uint32_t>(cbegin(), cend());
}

std::vector<uint32_t> VectorBase::as_vector() const {
    auto original_view = view();
    return std::vector<uint32_t>(original_view.begin(), original_view.end());
}

bool VectorBase::operator==(const VectorBase &other) const = default;

bool VectorBase::operator==(const std::vector<uint32_t> &other) const {
    auto original_view = view();
    return std::equal(original_view.begin(), original_view.end(), other.begin(), other.end());
}

uint32_t VectorBase::operator[](int32_t index) const {
    auto norm_index = normalized_index(index, m_original_size, m_value.size());
    return m_value[norm_index];
}

uint32_t& VectorBase::operator[](int32_t index) {
    auto norm_index = normalized_index(index, m_original_size, m_value.size());
    return m_value[norm_index];
}

}
