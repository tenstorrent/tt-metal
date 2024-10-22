// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "vector_base.hpp"

namespace tt::tt_metal {

namespace {
int32_t normalized_index(int32_t index, size_t container_size) {
    int32_t size = static_cast<int32_t>(container_size);

    if (index < 0) {
        index += size;
    }

    if (index < 0 || index >= size) {
        throw std::out_of_range("SimpleShape index out of range.");
    }

    return index;
}
}

bool VectorBase::operator==(const VectorBase &other) const {
    return this->m_value == other.m_value;
}

bool VectorBase::operator==(const std::vector<uint32_t> &other) const {
    return this->m_value == other;
}

uint32_t VectorBase::operator[](int32_t index) const {
    auto norm_index = normalized_index(index, m_value.size());
    return m_value[norm_index];
}

uint32_t& VectorBase::operator[](int32_t index) {
    auto norm_index = normalized_index(index, m_value.size());
    return m_value[norm_index];
}

}
