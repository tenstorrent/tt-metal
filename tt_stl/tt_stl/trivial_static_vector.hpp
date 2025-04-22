#include <array>
#include <type_traits>
#include <stdexcept>
#include <cstddef>
#include <algorithm>

namespace tt::stl {

template <typename T, std::size_t N>
class trivial_static_vector {
    static_assert(std::is_default_constructible_v<T>, "trivial_static_vector requires default constructible types");
    static_assert(std::is_copy_constructible_v<T>, "trivial_static_vector requires copy constructible types");
    static_assert(std::is_copy_assignable_v<T>, "trivial_static_vector requires copy assignable types");
    static_assert(
        std::is_nothrow_move_constructible_v<T>, "trivial_static_vector requires nothrow move constructible types");
    static_assert(std::is_nothrow_move_assignable_v<T>, "trivial_static_vector requires nothrow move assignable types");
    static_assert(std::is_nothrow_destructible_v<T>, "trivial_static_vector requires nothrow destructible types");

    std::array<T, N> data_{};
    size_t size_ = 0;

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    constexpr trivial_static_vector() = default;

    template <typename... Ts>
    constexpr trivial_static_vector(Ts... vals) {
        data_ = {vals...};
        size_ = sizeof...(Ts);
    }

    constexpr trivial_static_vector(const trivial_static_vector&) = default;
    constexpr trivial_static_vector(trivial_static_vector&&) noexcept = default;
    constexpr trivial_static_vector& operator=(const trivial_static_vector&) = default;
    constexpr trivial_static_vector& operator=(trivial_static_vector&&) noexcept = default;

    ~trivial_static_vector() noexcept = default;

    [[nodiscard]] constexpr reference at(size_type pos) {
        if (pos >= size_) {
            throw std::out_of_range("static_vector: index out of range");
        }
        return data_[pos];
    }

    [[nodiscard]] constexpr const_reference at(size_type pos) const {
        if (pos >= size_) {
            throw std::out_of_range("static_vector: index out of range");
        }
        return data_[pos];
    }

    [[nodiscard]] constexpr reference operator[](size_type pos) noexcept { return data_[pos]; }

    [[nodiscard]] constexpr const_reference operator[](size_type pos) const noexcept { return data_[pos]; }

    [[nodiscard]] constexpr reference front() noexcept { return data_[0]; }

    [[nodiscard]] constexpr const_reference front() const noexcept { return data_[0]; }

    [[nodiscard]] constexpr reference back() noexcept { return data_[size_ - 1]; }

    [[nodiscard]] constexpr const_reference back() const noexcept { return data_[size_ - 1]; }

    [[nodiscard]] constexpr pointer data() noexcept { return data_.data(); }

    [[nodiscard]] constexpr const_pointer data() const noexcept { return data_.data(); }

    [[nodiscard]] constexpr iterator begin() noexcept { return data_.data(); }

    [[nodiscard]] constexpr const_iterator begin() const noexcept { return data_.data(); }

    [[nodiscard]] constexpr const_iterator cbegin() const noexcept { return data_.data(); }

    [[nodiscard]] constexpr iterator end() noexcept { return data_.data() + size_; }

    [[nodiscard]] constexpr const_iterator end() const noexcept { return data_.data() + size_; }

    [[nodiscard]] constexpr const_iterator cend() const noexcept { return data_.data() + size_; }

    [[nodiscard]] constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

    [[nodiscard]] constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

    [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }

    [[nodiscard]] constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

    [[nodiscard]] constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

    [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }

    [[nodiscard]] constexpr size_type max_size() const noexcept { return N; }

    [[nodiscard]] constexpr size_type capacity() const noexcept { return N; }

    constexpr void clear() noexcept { size_ = 0; }

    template <typename... Args>
    constexpr reference emplace_back(Args&&... args) {
        if (size_ >= N) {
            throw std::length_error("static_vector: capacity exceeded");
        }

        data_[size_] = T(std::forward<Args>(args)...);
        ++size_;
        return data_[size_ - 1];
    }

    constexpr void push_back(const T& value) {
        if (size_ >= N) {
            throw std::length_error("static_vector: capacity exceeded");
        }

        data_[size_] = value;
        ++size_;
    }

    constexpr void push_back(T&& value) {
        if (size_ >= N) {
            throw std::length_error("static_vector: capacity exceeded");
        }

        data_[size_] = std::move(value);
        ++size_;
    }

    constexpr void pop_back() noexcept { --size_; }
};

template <typename T, std::size_t N>
constexpr bool operator==(const trivial_static_vector<T, N>& lhs, const trivial_static_vector<T, N>& rhs) noexcept {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    return std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0;
}

template <typename T, std::size_t N>
constexpr bool operator!=(const trivial_static_vector<T, N>& lhs, const trivial_static_vector<T, N>& rhs) noexcept {
    return !(lhs == rhs);
}

template <typename T, std::size_t N>
constexpr bool operator<(const trivial_static_vector<T, N>& lhs, const trivial_static_vector<T, N>& rhs) noexcept {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template <typename T, std::size_t N>
constexpr bool operator<=(const trivial_static_vector<T, N>& lhs, const trivial_static_vector<T, N>& rhs) noexcept {
    return !(rhs < lhs);
}

template <typename T, std::size_t N>
constexpr bool operator>(const trivial_static_vector<T, N>& lhs, const trivial_static_vector<T, N>& rhs) noexcept {
    return rhs < lhs;
}

template <typename T, std::size_t N>
constexpr bool operator>=(const trivial_static_vector<T, N>& lhs, const trivial_static_vector<T, N>& rhs) noexcept {
    return !(lhs < rhs);
}

}  // namespace tt::stl
