// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

// Container wrapper that allows negative indexing
class VectorBase {
public:
    VectorBase() = default;
    explicit VectorBase(const std::vector<uint32_t>& shape) : m_value(shape) {}
    explicit VectorBase(std::vector<uint32_t>&& shape) : m_value(std::move(shape)) {}
    explicit VectorBase(std::initializer_list<uint32_t> ilist) : m_value(ilist) {}
    template<std::size_t N>
    explicit VectorBase(const std::array<uint32_t, N>& arr) : m_value(arr.begin(), arr.end()) {}

    template<std::size_t N>
    bool operator==(const std::array<uint32_t, N> &other) const {
        bool same_size = m_value.size() == N;
        return same_size && std::equal(m_value.begin(), m_value.end(), other.begin());
    }

    bool operator==(const VectorBase &other) const;
    bool operator==(const std::vector<uint32_t> &other) const;

    uint32_t operator[](int32_t index) const;
    uint32_t &operator[](int32_t index);


    auto cbegin() const { return this->m_value.cbegin(); }
    auto cend() const { return this->m_value.cend(); }

    [[nodiscard]] const std::vector<uint32_t>& as_vector() const { return this->m_value; }

protected:
    std::vector<uint32_t> m_value;
};

}
