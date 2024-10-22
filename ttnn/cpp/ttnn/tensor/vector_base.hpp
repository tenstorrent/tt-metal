// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

// Container wrapper that allows negative indexing
class vector_base {
public:
    vector_base() = default;
    explicit vector_base(const std::vector<uint32_t>& shape) : mValue(shape) {}
    explicit vector_base(std::vector<uint32_t>&& shape) : mValue(std::move(shape)) {}
    explicit vector_base(std::initializer_list<uint32_t> ilist) : mValue(ilist) {}
    template<std::size_t N>
    explicit vector_base(const std::array<uint32_t, N>& arr) : mValue(arr.begin(), arr.end()) {}

    template<std::size_t N>
    bool operator==(const std::array<uint32_t, N> &other) const {
        bool sameSize = mValue.size() == N;
        return sameSize && std::equal(mValue.begin(), mValue.end(), other.begin());
    }

    bool operator==(const vector_base &other) const;
    bool operator==(const std::vector<uint32_t> &other) const;

    uint32_t operator[](int32_t index) const;
    uint32_t &operator[](int32_t index);


    auto cbegin() const { return this->mValue.cbegin(); }
    auto cend() const { return this->mValue.cend(); }

    [[nodiscard]] const std::vector<uint32_t>& as_vector() const { return this->mValue; }

protected:
    std::vector<uint32_t> mValue;
};

}
