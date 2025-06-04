// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace tt::stl {

// Most types are not typically aligned more than the size of a pointer. If in the future there is a special case to
// accommodate, these can be parameterized then.
inline constexpr std::size_t any_iterator_alignment = alignof(void*);
inline constexpr std::size_t any_range_alignment = alignof(void*);
// Typical size of an iterator is the size of a pointer. Default is size of two pointers to account for vtable pointer.
inline constexpr std::size_t default_any_iterator_capacity = 2 * sizeof(void*);
// Typical size of std::vector implementation is the size of three pointers. Default is the size of four pointers to
// account for vtable pointer.
inline constexpr std::size_t default_any_range_capacity = 4 * sizeof(void*);

struct input_range_tag {
    using iterator_category = std::input_iterator_tag;
};

struct forward_range_tag : input_range_tag {
    using base_tag = input_range_tag;
    using iterator_category = std::forward_iterator_tag;
};

struct bidirectional_range_tag : forward_range_tag {
    using base_tag = forward_range_tag;
    using iterator_category = std::bidirectional_iterator_tag;
};

struct random_access_range_tag : bidirectional_range_tag {
    using base_tag = bidirectional_range_tag;
    using iterator_category = std::random_access_iterator_tag;
};

struct sized_range_tag {};

struct sized_input_range_tag : input_range_tag, sized_range_tag {
    using base_tag = input_range_tag;
};

struct sized_forward_range_tag : forward_range_tag, sized_range_tag {
    using base_tag = forward_range_tag;
};

struct sized_bidirectional_range_tag : bidirectional_range_tag, sized_range_tag {
    using base_tag = bidirectional_range_tag;
};

struct sized_random_access_range_tag : random_access_range_tag, sized_range_tag {
    using base_tag = random_access_range_tag;
};

namespace detail {

// https://en.cppreference.com/w/cpp/concepts/destructible
template <class T>
inline constexpr bool destructible = std::is_nothrow_destructible_v<T>;

// https://en.cppreference.com/w/cpp/concepts/constructible_from
template <class T, class... TArgs>
inline constexpr bool constructible_from = destructible<T> and std::is_constructible_v<T, TArgs...>;

template <class TFrom, class TTo, class TEnable = void>
inline constexpr bool convertible_to = false;

// https://en.cppreference.com/w/cpp/concepts/convertible_to
template <class TFrom, class TTo>
inline constexpr bool convertible_to<TFrom, TTo, std::void_t<decltype(static_cast<TTo>(std::declval<TFrom>()))>> =
    std::is_convertible_v<TFrom, TTo>;

// https://en.cppreference.com/w/cpp/concepts/move_constructible
template <class T>
inline constexpr bool move_constructible = constructible_from<T, T> and convertible_to<T, T>;

// https://en.cppreference.com/w/cpp/concepts/copy_constructible
template <class T>
inline constexpr bool copy_constructible =
    move_constructible<T> and constructible_from<T, T&> and convertible_to<T&, T> and
    constructible_from<T, const T&> and convertible_to<const T&, T> and constructible_from<T, const T> and
    convertible_to<const T, T>;

// https://en.cppreference.com/w/cpp/types/remove_cvref
template <class T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// https://en.cppreference.com/w/cpp/ranges/iterator_t
template <class TRange>
using iterator_t = decltype(std::begin(std::declval<TRange&>()));

template <class TRange>
using sentinel_t = decltype(std::end(std::declval<TRange&>()));

// https://en.cppreference.com/w/cpp/iterator/iter_t
template <class TIterator>
using iter_reference_t = decltype(*std::declval<TIterator&>());

// https://en.cppreference.com/w/cpp/ranges/common_range
template <class TRange>
inline constexpr bool common_range = std::is_same_v<iterator_t<TRange>, sentinel_t<TRange>>;

template <class TRange, class TEnable = void>
inline constexpr bool sized_range = false;

// https://en.cppreference.com/w/cpp/ranges/sized_range
template <class TRange>
inline constexpr bool sized_range<TRange, std::void_t<decltype(std::size(std::declval<TRange&>()))>> = true;

template <class T>
using base_tag_t = typename T::base_tag;

template <class T>
using iterator_category_t = typename T::iterator_category;

template <class T, std::size_t Align>
inline constexpr bool is_aligned_to_v = Align % alignof(T) == 0;

template <class TIterator, class TReference, class TIteratorTag, class TEnable = void>
inline constexpr bool iterator_compatible = false;

template <class TIterator, class TReference, class TIteratorTag>
inline constexpr bool iterator_compatible<
    TIterator,
    TReference,
    TIteratorTag,
    std::void_t<iter_reference_t<TIterator>, iterator_category_t<std::iterator_traits<TIterator>>>> =
    convertible_to<iter_reference_t<TIterator>, TReference> and
    std::is_base_of_v<TIteratorTag, iterator_category_t<std::iterator_traits<TIterator>>>;

template <class TRange, class TReference, class TRangeTag, class TEnable = void>
inline constexpr bool range_compatible =
    copy_constructible<TRange> and common_range<TRange> and
    iterator_compatible<iterator_t<TRange>, TReference, iterator_category_t<TRangeTag>>;

template <class TRange, class TReference, class TRangeTag>
inline constexpr bool
    range_compatible<TRange, TReference, TRangeTag, std::enable_if_t<std::is_base_of_v<sized_range_tag, TRangeTag>>> =
        range_compatible<TRange, TReference, base_tag_t<TRangeTag>> and sized_range<TRange>;

template <class TRangeTag, class TCanonicalTag, class TEnable = void>
inline constexpr bool is_tag_v = std::is_base_of_v<TCanonicalTag, TRangeTag>;

template <class TRangeTag, class TCanonicalTag>
inline constexpr bool is_tag_v<TRangeTag, TCanonicalTag, std::void_t<base_tag_t<TRangeTag>>> =
    std::is_base_of_v<TCanonicalTag, TRangeTag> and not std::is_base_of_v<TCanonicalTag, base_tag_t<TRangeTag>>;

template <class TPointer>
using pointer_dereference_t = decltype(std::declval<const TPointer&>().operator->());

template <class TPointer, class TEnable = void>
struct address {};

template <class TPointer>
struct address<TPointer, std::void_t<pointer_dereference_t<TPointer>>> : address<pointer_dereference_t<TPointer>> {};

template <class TValue>
struct address<TValue*> {
    using type = TValue*;
};

template <class TPointer>
using address_t = typename address<TPointer>::type;

// https://en.cppreference.com/w/cpp/memory/to_address
template <class TValue>
constexpr TValue* to_address(TValue* ptr) noexcept {
    return ptr;
}

template <class TPointer>
constexpr address_t<TPointer> to_address(const TPointer& ptr) noexcept {
    return detail::to_address(ptr.operator->());
}

template <class TReference>
using addressof_t = decltype(detail::to_address(&std::declval<TReference>()));

template <class TReference, class TIteratorTag, class TEnable = void>
struct AnyIteratorTraits {
    using difference_type = std::ptrdiff_t;
    using value_type = detail::remove_cvref_t<TReference>;
    using pointer = void;
    using reference = TReference;
    using iterator_category = TIteratorTag;
};

template <class TReference, class TIteratorTag>
struct AnyIteratorTraits<TReference, TIteratorTag, std::void_t<addressof_t<TReference>>> {
    using difference_type = std::ptrdiff_t;
    using value_type = detail::remove_cvref_t<TReference>;
    using pointer = addressof_t<TReference>;
    using reference = TReference;
    using iterator_category = TIteratorTag;
};

class ICopyConstructible {
public:
    virtual void uninitialized_copy_to(void* dst_ptr) const = 0;
    virtual void uninitialized_move_to(void* dst_ptr) noexcept = 0;
    virtual ~ICopyConstructible() noexcept = default;
};

template <class TAnyIterator, class = iterator_category_t<TAnyIterator>>
class IAnyIteratorAdaptor;

template <class TAnyIterator>
class IAnyIteratorAdaptor<TAnyIterator, std::input_iterator_tag> : public ICopyConstructible {
public:
    [[nodiscard]] virtual const std::type_info& type() const noexcept = 0;
    [[nodiscard]] virtual bool operator==(const IAnyIteratorAdaptor& other) const = 0;
    [[nodiscard]] virtual typename TAnyIterator::reference operator*() const = 0;
    virtual void operator++() = 0;
};

template <class TAnyIterator>
class IAnyIteratorAdaptor<TAnyIterator, std::forward_iterator_tag>
    : public IAnyIteratorAdaptor<TAnyIterator, std::input_iterator_tag> {
public:
    using IAnyIteratorAdaptor<TAnyIterator, std::input_iterator_tag>::operator++;
    [[nodiscard]] virtual TAnyIterator operator++(int) = 0;
};

template <class TAnyIterator>
class IAnyIteratorAdaptor<TAnyIterator, std::bidirectional_iterator_tag>
    : public IAnyIteratorAdaptor<TAnyIterator, std::forward_iterator_tag> {
public:
    virtual void operator--() = 0;
    [[nodiscard]] virtual TAnyIterator operator--(int) = 0;
};

template <class TAnyIterator>
class IAnyIteratorAdaptor<TAnyIterator, std::random_access_iterator_tag>
    : public IAnyIteratorAdaptor<TAnyIterator, std::bidirectional_iterator_tag> {
public:
    virtual void operator+=(typename TAnyIterator::difference_type) = 0;
    [[nodiscard]] virtual TAnyIterator operator+(typename TAnyIterator::difference_type) const = 0;
    [[nodiscard]] virtual typename TAnyIterator::difference_type operator-(const IAnyIteratorAdaptor&) const = 0;
    [[nodiscard]] virtual typename TAnyIterator::reference operator[](typename TAnyIterator::difference_type) const = 0;
};

template <class T>
class Final final : public T {
    using T::T;
};

template <class TAnyIterator, class TIterator, class TIteratorTag = iterator_category_t<TAnyIterator>>
class AnyIteratorAdaptor;

template <class TAnyIterator, class TIterator>
class AnyIteratorAdaptor<TAnyIterator, TIterator, std::input_iterator_tag> : public IAnyIteratorAdaptor<TAnyIterator> {
    using final_iterator_adaptor_type = Final<AnyIteratorAdaptor<TAnyIterator, TIterator>>;

protected:
    TIterator iterator;

public:
    AnyIteratorAdaptor(const AnyIteratorAdaptor&) = default;
    AnyIteratorAdaptor(AnyIteratorAdaptor&&) noexcept = default;

    template <class UIterator, class = std::enable_if_t<std::is_same_v<remove_cvref_t<UIterator>, TIterator>>>
    AnyIteratorAdaptor(UIterator&& iterator) : iterator(static_cast<UIterator&&>(iterator)) {}

    void uninitialized_copy_to(void* dst_ptr) const override {
        ::new (dst_ptr) final_iterator_adaptor_type(static_cast<const final_iterator_adaptor_type&>(*this));
    }

    void uninitialized_move_to(void* dst_ptr) noexcept override {
        ::new (dst_ptr) final_iterator_adaptor_type(static_cast<final_iterator_adaptor_type&&>(*this));
    }

    [[nodiscard]] const std::type_info& type() const noexcept override { return typeid(AnyIteratorAdaptor); }

    [[nodiscard]] bool operator==(
        const IAnyIteratorAdaptor<TAnyIterator, std::input_iterator_tag>& other) const override {
        return type() == other.type() and iterator == static_cast<const AnyIteratorAdaptor&>(other).iterator;
    }

    [[nodiscard]] typename TAnyIterator::reference operator*() const override {
        return static_cast<typename TAnyIterator::reference>(*iterator);
    }

    void operator++() override { ++iterator; }
};

template <class TAnyIterator, class TIterator>
class AnyIteratorAdaptor<TAnyIterator, TIterator, std::forward_iterator_tag>
    : public AnyIteratorAdaptor<TAnyIterator, TIterator, std::input_iterator_tag> {
public:
    using AnyIteratorAdaptor<TAnyIterator, TIterator, std::input_iterator_tag>::AnyIteratorAdaptor;

    [[nodiscard]] TAnyIterator operator++(int) override { return TAnyIterator(this->iterator++); }
};

template <class TAnyIterator, class TIterator>
class AnyIteratorAdaptor<TAnyIterator, TIterator, std::bidirectional_iterator_tag>
    : public AnyIteratorAdaptor<TAnyIterator, TIterator, std::forward_iterator_tag> {
public:
    using AnyIteratorAdaptor<TAnyIterator, TIterator, std::forward_iterator_tag>::AnyIteratorAdaptor;

    void operator--() override { --this->iterator; }

    [[nodiscard]] TAnyIterator operator--(int) override { return TAnyIterator(this->iterator--); }
};

template <class TAnyIterator, class TIterator>
class AnyIteratorAdaptor<TAnyIterator, TIterator, std::random_access_iterator_tag>
    : public AnyIteratorAdaptor<TAnyIterator, TIterator, std::bidirectional_iterator_tag> {
public:
    using AnyIteratorAdaptor<TAnyIterator, TIterator, std::bidirectional_iterator_tag>::AnyIteratorAdaptor;

    void operator+=(typename TAnyIterator::difference_type offset) override { this->iterator += offset; }

    [[nodiscard]] TAnyIterator operator+(typename TAnyIterator::difference_type offset) const override {
        return TAnyIterator(this->iterator + offset);
    }

    [[nodiscard]] typename TAnyIterator::difference_type operator-(
        const IAnyIteratorAdaptor<TAnyIterator>& other) const override {
        return this->type() == other.type() ? this->iterator - static_cast<const AnyIteratorAdaptor&>(other).iterator
                                            : this->iterator - TIterator{};
    }

    [[nodiscard]] typename TAnyIterator::reference operator[](
        typename TAnyIterator::difference_type offset) const override {
        return static_cast<typename TAnyIterator::reference>(this->iterator[offset]);
    }
};

}  // namespace detail

template <class TReference, class TIteratorTag, std::size_t Capacity = default_any_iterator_capacity>
class AnyIterator : public detail::AnyIteratorTraits<TReference, TIteratorTag> {
    using traits = detail::AnyIteratorTraits<TReference, TIteratorTag>;

public:
    using typename traits::difference_type;
    using typename traits::iterator_category;
    using typename traits::pointer;
    using typename traits::reference;
    using typename traits::value_type;

private:
    alignas(any_iterator_alignment) std::byte bytes[Capacity];

    using iterator_adaptor_type = detail::IAnyIteratorAdaptor<AnyIterator>;

    iterator_adaptor_type& iterator_adaptor() noexcept {
        return *static_cast<iterator_adaptor_type*>(static_cast<void*>(bytes));
    }

    const iterator_adaptor_type& iterator_adaptor() const noexcept {
        return *static_cast<const iterator_adaptor_type*>(static_cast<const void*>(bytes));
    }

public:
    AnyIterator(const AnyIterator& other) { other.iterator_adaptor().uninitialized_copy_to(bytes); }

    AnyIterator(AnyIterator&& other) noexcept { other.iterator_adaptor().uninitialized_move_to(bytes); }

private:
    template <class TIterator, class TEnable = void>
    struct enable_if_iterator_compatible : std::enable_if<
                                               detail::iterator_compatible<TIterator, reference, iterator_category>,
                                               detail::Final<detail::AnyIteratorAdaptor<AnyIterator, TIterator>>> {};

    // Don't hide copy and move constructors
    template <class TIterator>
    struct enable_if_iterator_compatible<TIterator, std::enable_if_t<std::is_same_v<TIterator, AnyIterator>>> {};

    // Discard candidate for implicit construction from adaptor type
    template <class TIterator>
    struct enable_if_iterator_compatible<
        TIterator,
        std::enable_if_t<std::is_same_v<TIterator, iterator_adaptor_type>>> {};

    template <class TAdaptor>
    struct enable_if_storage_compatible
        : std::enable_if<
              sizeof(AnyIterator) >= sizeof(TAdaptor) and detail::is_aligned_to_v<TAdaptor, alignof(AnyIterator)>,
              TAdaptor> {};

public:
    template <
        class TIterator,
        class TAdaptor = typename enable_if_iterator_compatible<detail::remove_cvref_t<TIterator>>::type,
        class = typename enable_if_storage_compatible<TAdaptor>::type>
    // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
    AnyIterator(TIterator&& iterator) {
        ::new (bytes) TAdaptor(static_cast<TIterator&&>(iterator));
    }

    void swap(AnyIterator& rhs) noexcept {
        AnyIterator tmp(static_cast<AnyIterator&&>(rhs));
        rhs.iterator_adaptor().~iterator_adaptor_type();
        iterator_adaptor().uninitialized_move_to(rhs.bytes);
        iterator_adaptor().~iterator_adaptor_type();
        tmp.iterator_adaptor().uninitialized_move_to(bytes);
    }

    AnyIterator& operator=(const AnyIterator& other) {
        if (this != &other) {
            iterator_adaptor().~iterator_adaptor_type();
            other.iterator_adaptor().uninitialized_copy_to(bytes);
        }
        return *this;
    }

    AnyIterator& operator=(AnyIterator&& other) noexcept {
        if (this != &other) {
            iterator_adaptor().~iterator_adaptor_type();
            other.iterator_adaptor().uninitialized_move_to(bytes);
        }
        return *this;
    }

    ~AnyIterator() noexcept { iterator_adaptor().~iterator_adaptor_type(); }

    // input methods
    static_assert(std::is_base_of_v<std::input_iterator_tag, iterator_category>);

    [[nodiscard]] friend bool operator==(const AnyIterator& lhs, const AnyIterator& rhs) {
        return lhs.iterator_adaptor() == rhs.iterator_adaptor();
    }

    [[nodiscard]] friend bool operator!=(const AnyIterator& lhs, const AnyIterator& rhs) { return not(lhs == rhs); }

    [[nodiscard]] reference operator*() const { return *iterator_adaptor(); }

private:
    struct enable_if_addressable : std::enable_if<not std::is_void_v<pointer>, enable_if_addressable> {};

public:
    template <class TEnable = enable_if_addressable, class = typename TEnable::type>
    [[nodiscard]] pointer operator->() const {
        static_assert(std::is_same_v<TEnable, enable_if_addressable>);
        return detail::to_address(&*iterator_adaptor());
    }

    AnyIterator& operator++() {
        ++iterator_adaptor();
        return *this;
    }

    // forward methods
private:
    struct enable_if_forward
        : std::enable_if<std::is_base_of_v<std::forward_iterator_tag, iterator_category>, enable_if_forward> {};

public:
    template <class TEnable = enable_if_forward, class = typename TEnable::type>
    AnyIterator() noexcept : AnyIterator(static_cast<std::add_pointer_t<reference>>(nullptr)) {}

    template <class TEnable = enable_if_forward, class = typename TEnable::type>
    [[nodiscard]] AnyIterator operator++(int) {
        static_assert(std::is_same_v<TEnable, enable_if_forward>);
        return iterator_adaptor()++;
    }

    // bidirectional methods
private:
    struct enable_if_bidirectional : std::enable_if<
                                         std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>,
                                         enable_if_bidirectional> {};

public:
    template <class TEnable = enable_if_bidirectional, class = typename TEnable::type>
    AnyIterator& operator--() {
        static_assert(std::is_same_v<TEnable, enable_if_bidirectional>);
        --iterator_adaptor();
        return *this;
    }

    template <class TEnable = enable_if_bidirectional, class = typename TEnable::type>
    [[nodiscard]] AnyIterator operator--(int) {
        static_assert(std::is_same_v<TEnable, enable_if_bidirectional>);
        return iterator_adaptor()--;
    }

    // random access methods
private:
    struct enable_if_random_access : std::enable_if<
                                         std::is_base_of_v<std::random_access_iterator_tag, iterator_category>,
                                         enable_if_random_access> {};

public:
    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    AnyIterator& operator+=(difference_type offset) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        iterator_adaptor() += offset;
        return *this;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    AnyIterator& operator-=(difference_type offset) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        iterator_adaptor() += -offset;
        return *this;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] AnyIterator operator+(difference_type offset) const {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return iterator_adaptor() + offset;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] friend AnyIterator operator+(difference_type offset, const AnyIterator& other) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return other.iterator_adaptor() + offset;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] AnyIterator operator-(difference_type offset) const {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return iterator_adaptor() + -offset;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] difference_type operator-(const AnyIterator& other) const {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return iterator_adaptor() - other.iterator_adaptor();
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] reference operator[](difference_type offset) const {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return iterator_adaptor()[offset];
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] friend bool operator<(const AnyIterator& lhs, const AnyIterator& rhs) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return (lhs - rhs) < 0;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] friend bool operator>(const AnyIterator& lhs, const AnyIterator& rhs) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return (lhs - rhs) > 0;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] friend bool operator<=(const AnyIterator& lhs, const AnyIterator& rhs) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return (lhs - rhs) <= 0;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] friend bool operator>=(const AnyIterator& lhs, const AnyIterator& rhs) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return (lhs - rhs) >= 0;
    }
};

template <class TReference, class TIteratorTag, std::size_t Capacity>
void swap(
    AnyIterator<TReference, TIteratorTag, Capacity>& lhs,
    AnyIterator<TReference, TIteratorTag, Capacity>& rhs) noexcept {
    lhs.swap(rhs);
}

namespace detail {

template <class TReference, class TRangeTag, class TIterator, class TEnable = void>
struct AnyRangeTraits : AnyIteratorTraits<TReference, iterator_category_t<TRangeTag>> {
    using size_type = std::size_t;
    using iterator = TIterator;
};

template <class TReference, class TRangeTag, class TIterator>
struct AnyRangeTraits<
    TReference,
    TRangeTag,
    TIterator,
    std::enable_if_t<std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category_t<TRangeTag>>>>
    : AnyRangeTraits<TReference, forward_range_tag, TIterator> {
    using reverse_iterator = std::reverse_iterator<typename AnyRangeTraits::iterator>;
};

template <class TAnyRange, class TRangeTag = typename TAnyRange::range_category, class TEnable = void>
class IAnyRangeAdaptor;

template <class TAnyRange, class TRangeTag>
class IAnyRangeAdaptor<TAnyRange, TRangeTag, std::enable_if_t<is_tag_v<TRangeTag, input_range_tag>>>
    : public ICopyConstructible {
public:
    [[nodiscard]] virtual typename TAnyRange::iterator begin() = 0;
    [[nodiscard]] virtual typename TAnyRange::iterator end() = 0;
};

template <class TAnyRange, class TRangeTag>
class IAnyRangeAdaptor<TAnyRange, TRangeTag, std::enable_if_t<is_tag_v<TRangeTag, forward_range_tag>>>
    : public IAnyRangeAdaptor<TAnyRange, input_range_tag> {};

template <class TAnyRange, class TRangeTag>
class IAnyRangeAdaptor<TAnyRange, TRangeTag, std::enable_if_t<is_tag_v<TRangeTag, bidirectional_range_tag>>>
    : public IAnyRangeAdaptor<TAnyRange, forward_range_tag> {
public:
    [[nodiscard]] virtual typename TAnyRange::reverse_iterator rbegin() = 0;
    [[nodiscard]] virtual typename TAnyRange::reverse_iterator rend() = 0;
};

template <class TAnyRange, class TRangeTag>
class IAnyRangeAdaptor<TAnyRange, TRangeTag, std::enable_if_t<is_tag_v<TRangeTag, random_access_range_tag>>>
    : public IAnyRangeAdaptor<TAnyRange, bidirectional_range_tag> {
public:
    [[nodiscard]] virtual typename TAnyRange::reference operator[](typename TAnyRange::size_type) = 0;
};

template <class TAnyRange, class TRangeTag>
class IAnyRangeAdaptor<TAnyRange, TRangeTag, std::enable_if_t<is_tag_v<TRangeTag, sized_range_tag>>>
    : public IAnyRangeAdaptor<TAnyRange, base_tag_t<TRangeTag>> {
public:
    [[nodiscard]] virtual typename TAnyRange::size_type size() const = 0;
};

template <class TAnyRange, class TRange, class TRangeTag = typename TAnyRange::range_category, class TEnable = void>
class AnyRangeAdaptor;

template <class TAnyRange, class TRange>
class AnyRangeAdaptor<TAnyRange, TRange, input_range_tag> : public IAnyRangeAdaptor<TAnyRange> {
    using final_range_adaptor_type = Final<AnyRangeAdaptor<TAnyRange, TRange>>;

protected:
    TRange range;

public:
    AnyRangeAdaptor(const AnyRangeAdaptor&) = default;
    AnyRangeAdaptor(AnyRangeAdaptor&&) noexcept = default;

    template <class URange, class = std::enable_if_t<std::is_same_v<remove_cvref_t<TRange>, remove_cvref_t<URange>>>>
    AnyRangeAdaptor(URange&& range) : range(static_cast<URange&&>(range)) {}

    void uninitialized_copy_to(void* dst_ptr) const override {
        ::new (dst_ptr) final_range_adaptor_type(static_cast<const final_range_adaptor_type&>(*this));
    }

    void uninitialized_move_to(void* dst_ptr) noexcept override {
        ::new (dst_ptr) final_range_adaptor_type(static_cast<final_range_adaptor_type&&>(*this));
    }

    [[nodiscard]] typename TAnyRange::iterator begin() override { return range.begin(); }

    [[nodiscard]] typename TAnyRange::iterator end() override { return range.end(); }
};

template <class TAnyRange, class TRange, class TRangeTag>
class AnyRangeAdaptor<TAnyRange, TRange, TRangeTag, std::enable_if_t<is_tag_v<TRangeTag, forward_range_tag>>>
    : public AnyRangeAdaptor<TAnyRange, TRange, input_range_tag> {
public:
    using AnyRangeAdaptor<TAnyRange, TRange, input_range_tag>::AnyRangeAdaptor;
};

template <class TAnyRange, class TRange, class TRangeTag>
class AnyRangeAdaptor<TAnyRange, TRange, TRangeTag, std::enable_if_t<is_tag_v<TRangeTag, bidirectional_range_tag>>>
    : public AnyRangeAdaptor<TAnyRange, TRange, forward_range_tag> {
public:
    using AnyRangeAdaptor<TAnyRange, TRange, forward_range_tag>::AnyRangeAdaptor;

    [[nodiscard]] typename TAnyRange::reverse_iterator rbegin() override { return this->range.rbegin(); }

    [[nodiscard]] typename TAnyRange::reverse_iterator rend() override { return this->range.rend(); }
};

template <class TAnyRange, class TRange, class TRangeTag>
class AnyRangeAdaptor<TAnyRange, TRange, TRangeTag, std::enable_if_t<is_tag_v<TRangeTag, random_access_range_tag>>>
    : public AnyRangeAdaptor<TAnyRange, TRange, bidirectional_range_tag> {
public:
    using AnyRangeAdaptor<TAnyRange, TRange, bidirectional_range_tag>::AnyRangeAdaptor;

    [[nodiscard]] typename TAnyRange::reference operator[](typename TAnyRange::size_type index) override {
        return this->range[index];
    }
};

template <class TAnyRange, class TRange, class TRangeTag>
class AnyRangeAdaptor<TAnyRange, TRange, TRangeTag, std::enable_if_t<is_tag_v<TRangeTag, sized_range_tag>>>
    : public AnyRangeAdaptor<TAnyRange, TRange, base_tag_t<TRangeTag>> {
public:
    using AnyRangeAdaptor<TAnyRange, TRange, base_tag_t<TRangeTag>>::AnyRangeAdaptor;

    [[nodiscard]] typename TAnyRange::size_type size() const override { return this->range.size(); }
};

}  // namespace detail

/**
 * @brief Container for type-erasing ranges to a common range interface.
 * @tparam TReference The reference type for the common range interface.
 * @tparam TRangeTag A tag type that indicates the iterator category of the common iterator interface and whether the
 * common range interface should include size-related methods.
 * @tparam TAnyIterator The common iterator interface to use.
 * @tparam Capacity The number of bytes in the common range interface class layout for storing type-erased ranges.
 */
template <
    class TReference,
    class TRangeTag,
    class TAnyIterator = AnyIterator<TReference, detail::iterator_category_t<TRangeTag>>,
    std::size_t Capacity = default_any_range_capacity>
class BasicAnyRange : public detail::AnyRangeTraits<TReference, TRangeTag, TAnyIterator> {
    using traits = detail::AnyRangeTraits<TReference, TRangeTag, TAnyIterator>;

public:
    using range_category = TRangeTag;
    using typename traits::difference_type;
    using typename traits::iterator;
    using typename traits::reference;
    using typename traits::size_type;
    using typename traits::value_type;

private:
    alignas(any_range_alignment) mutable std::byte bytes[Capacity];

    using range_adaptor_type = detail::IAnyRangeAdaptor<BasicAnyRange, range_category>;

    range_adaptor_type& range_adaptor() const noexcept {
        return *static_cast<range_adaptor_type*>(static_cast<void*>(bytes));
    }

public:
    BasicAnyRange() = delete;

    BasicAnyRange(const BasicAnyRange& other) { other.range_adaptor().uninitialized_copy_to(bytes); }

    BasicAnyRange(BasicAnyRange&& other) noexcept { other.range_adaptor().uninitialized_move_to(bytes); }

private:
    template <class TRange, class TEnable = void>
    struct enable_if_range_compatible : std::enable_if<
                                            detail::range_compatible<TRange, reference, range_category>,
                                            detail::Final<detail::AnyRangeAdaptor<BasicAnyRange, TRange>>> {};

    // Don't hide copy and move constructors
    template <class TRange>
    struct enable_if_range_compatible<
        TRange,
        std::enable_if_t<std::is_same_v<detail::remove_cvref_t<TRange>, BasicAnyRange>>> {};

    template <class TAdaptor>
    struct enable_if_storage_compatible
        : std::enable_if<
              sizeof(BasicAnyRange) >= sizeof(TAdaptor) and detail::is_aligned_to_v<TAdaptor, alignof(BasicAnyRange)>,
              TAdaptor> {};

    template <class TIterator>
    struct enable_if_iterator_compatible : std::enable_if<std::is_constructible_v<iterator, TIterator>> {};

public:
    /**
     * @brief Constructor from forwarding reference of underlying range type
     * @param range The cvref-qualified range type to erase; lvalues are stored as reference, rvalues are
     * move-constructed into storage, transferring ownership.
     */
    template <
        class TRange,
        class TAdaptor = typename enable_if_range_compatible<TRange>::type,
        class = typename enable_if_storage_compatible<TAdaptor>::type,
        class = typename enable_if_iterator_compatible<detail::iterator_t<TRange>>::type>
    // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
    BasicAnyRange(TRange&& range) {
        ::new (static_cast<void*>(bytes)) TAdaptor(static_cast<TRange&&>(range));
    }

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(clang::lifetimebound)

    /**
     * @brief Higher-priority constructor from lvalue reference of underlying range type
     * @note Diagnoses attempts to type-erase a dangling reference if lifetimebound attribute is supported.
     * @param range The lvalue reference range type to erase; stored as reference.
     */
    template <
        class TRange,
        class TAdaptor = typename enable_if_range_compatible<TRange&>::type,
        class = typename enable_if_storage_compatible<TAdaptor>::type,
        class = typename enable_if_iterator_compatible<detail::iterator_t<TRange>>::type>
    BasicAnyRange(TRange& range [[clang::lifetimebound]]) {
        ::new (static_cast<void*>(bytes)) TAdaptor(range);
    }

#endif
#endif

    void swap(BasicAnyRange& rhs) noexcept {
        BasicAnyRange tmp(static_cast<BasicAnyRange&&>(rhs));
        rhs.range_adaptor().~range_adaptor_type();
        range_adaptor().uninitialized_move_to(rhs.bytes);
        range_adaptor().~range_adaptor_type();
        tmp.range_adaptor().uninitialized_move_to(bytes);
    }

    BasicAnyRange& operator=(const BasicAnyRange& other) {
        if (this != &other) {
            range_adaptor().~range_adaptor_type();
            other.range_adaptor().uninitialized_copy_to(bytes);
        }
        return *this;
    }

    BasicAnyRange& operator=(BasicAnyRange&& other) noexcept {
        if (this != &other) {
            range_adaptor().~range_adaptor_type();
            other.range_adaptor().uninitialized_move_to(bytes);
        }
        return *this;
    }

    ~BasicAnyRange() noexcept { range_adaptor().~range_adaptor_type(); }

    // input methods
    static_assert(std::is_base_of_v<input_range_tag, range_category>);

    [[nodiscard]] iterator begin() const { return range_adaptor().begin(); }

    [[nodiscard]] iterator end() const { return range_adaptor().end(); }

    // bidirectional methods

    template <class TRangeTraits = traits>
    [[nodiscard]] typename TRangeTraits::reverse_iterator rbegin() const {
        static_assert(std::is_same_v<TRangeTraits, traits>);
        return range_adaptor().rbegin();
    }

    template <class TRangeTraits = traits>
    [[nodiscard]] typename TRangeTraits::reverse_iterator rend() const {
        static_assert(std::is_same_v<TRangeTraits, traits>);
        return range_adaptor().rend();
    }

    // sized methods
private:
    struct enable_if_sized : std::enable_if<detail::sized_range<range_adaptor_type>, traits> {};

public:
    template <class TEnable = enable_if_sized, class = typename TEnable::type>
    [[nodiscard]] size_type size() const {
        static_assert(std::is_same_v<TEnable, enable_if_sized>);
        return range_adaptor().size();
    }

    template <class TEnable = enable_if_sized, class = typename TEnable::type>
    [[nodiscard]] bool empty() const {
        static_assert(std::is_same_v<TEnable, enable_if_sized>);
        return range_adaptor().size() == 0;
    }

    // random access methods
private:
    struct enable_if_random_access
        : std::enable_if<std::is_base_of_v<random_access_range_tag, range_category>, traits> {};

public:
    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] reference operator[](size_type index) const {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return range_adaptor()[index];
    }
};

template <class TReference, class TRangeTag, class TAnyIterator, std::size_t Capacity>
void swap(
    BasicAnyRange<TReference, TRangeTag, TAnyIterator, Capacity>& lhs,
    BasicAnyRange<TReference, TRangeTag, TAnyIterator, Capacity>& rhs) noexcept {
    lhs.swap(rhs);
}

template <
    class TReference,
    class TRangeTag,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyRange = BasicAnyRange<
    TReference,
    TRangeTag,
    AnyIterator<TReference, detail::iterator_category_t<TRangeTag>, IteratorCapacity>,
    RangeCapacity>;

template <
    class TReference,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyInputRange = AnyRange<TReference, input_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TReference,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyForwardRange = AnyRange<TReference, forward_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TReference,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyBidirectionalRange = AnyRange<TReference, bidirectional_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TReference,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyRandomAccessRange = AnyRange<TReference, random_access_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TReference,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnySizedInputRange = AnyRange<TReference, sized_input_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TReference,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnySizedForwardRange = AnyRange<TReference, sized_forward_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TReference,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnySizedBidirectionalRange = AnyRange<TReference, sized_bidirectional_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TReference,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnySizedRandomAccessRange = AnyRange<TReference, sized_random_access_range_tag, RangeCapacity, IteratorCapacity>;

template <class TReference, class TRangeTag, class... TRanges>
using AnyRangeFor = AnyRange<
    // cause substitution failure if any type in pack does not satisfy alignment requirement for range or iterator
    std::enable_if_t<(... and detail::is_aligned_to_v<TRanges, any_range_alignment>), TReference>,
    std::enable_if_t<(... and detail::is_aligned_to_v<detail::iterator_t<TRanges>, any_iterator_alignment>), TRangeTag>,
    // add sizeof vtable pointer
    sizeof(void*) + std::max({sizeof(TRanges)...}),
    sizeof(void*) + std::max({sizeof(detail::iterator_t<TRanges>)...})>;

template <class TReference, class... TRanges>
using AnyInputRangeFor = AnyRangeFor<TReference, input_range_tag, TRanges...>;

template <class TReference, class... TRanges>
using AnyForwardRangeFor = AnyRangeFor<TReference, forward_range_tag, TRanges...>;

template <class TReference, class... TRanges>
using AnyBidirectionalRangeFor = AnyRangeFor<TReference, bidirectional_range_tag, TRanges...>;

template <class TReference, class... TRanges>
using AnyRandomAccessRangeFor = AnyRangeFor<TReference, random_access_range_tag, TRanges...>;

template <class TReference, class... TRanges>
using AnySizedInputRangeFor = AnyRangeFor<TReference, sized_input_range_tag, TRanges...>;

template <class TReference, class... TRanges>
using AnySizedForwardRangeFor = AnyRangeFor<TReference, sized_forward_range_tag, TRanges...>;

template <class TReference, class... TRanges>
using AnySizedBidirectionalRangeFor = AnyRangeFor<TReference, sized_bidirectional_range_tag, TRanges...>;

template <class TReference, class... TRanges>
using AnySizedRandomAccessRangeFor = AnyRangeFor<TReference, sized_random_access_range_tag, TRanges...>;

#define MAKE_ANY_RANGE(NAME, ...)         \
    class NAME : public __VA_ARGS__ {     \
        using __VA_ARGS__::BasicAnyRange; \
    }

}  // namespace tt::stl
