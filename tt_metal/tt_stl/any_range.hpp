// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace tt::stl {

inline constexpr std::size_t any_iterator_alignment = alignof(void *);
inline constexpr std::size_t any_range_alignment = alignof(void *);
inline constexpr std::size_t default_any_iterator_capacity = 16;
inline constexpr std::size_t default_any_range_capacity = 32;

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

template <class T>
inline constexpr bool destructible = std::is_nothrow_destructible_v<T>;

template <class T, class... TArgs>
inline constexpr bool constructible_from = destructible<T> and std::is_constructible_v<T, TArgs...>;

template <class TFrom, class TTo, class TEnable = void>
inline constexpr bool convertible_to = false;

template <class TFrom, class TTo>
inline constexpr bool convertible_to<TFrom, TTo, std::void_t<decltype(static_cast<TTo>(std::declval<TFrom>()))>> =
    std::is_convertible_v<TFrom, TTo>;

template <class T>
inline constexpr bool move_constructible = constructible_from<T, T> and convertible_to<T, T>;

template <class T>
inline constexpr bool copy_constructible =
    move_constructible<T> and constructible_from<T, T &> and convertible_to<T &, T> and
    constructible_from<T, const T &> and convertible_to<const T &, T> and constructible_from<T, const T> and
    convertible_to<const T, T>;

template <class TRange>
using iterator_t = decltype(std::begin(std::declval<TRange &>()));

template <class TRange>
using sentinel_t = decltype(std::end(std::declval<TRange &>()));

template <class TIterator>
using iter_reference_t = decltype(*std::declval<TIterator &>());

template <class TRange>
inline constexpr bool common_range = std::is_same_v<iterator_t<TRange>, sentinel_t<TRange>>;

template <class TRange, class TEnable = void>
inline constexpr bool sized_range = false;

template <class TRange>
inline constexpr bool sized_range<TRange, std::void_t<decltype(std::size(std::declval<TRange &>()))>> = true;

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

template <class T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class TIterator, class TReference, class TIteratorTag>
inline constexpr bool iterator_compatible_cvref =
    iterator_compatible<remove_cvref_t<TIterator>, TReference, TIteratorTag>;

template <class TRangeTag, class TCanonicalTag, class TEnable = void>
inline constexpr bool is_tag_v = std::is_base_of_v<TCanonicalTag, TRangeTag>;

template <class TRangeTag, class TCanonicalTag>
inline constexpr bool is_tag_v<TRangeTag, TCanonicalTag, std::void_t<base_tag_t<TRangeTag>>> =
    std::is_base_of_v<TCanonicalTag, TRangeTag> and not std::is_base_of_v<TCanonicalTag, base_tag_t<TRangeTag>>;

struct type_info {};

template <class TValue>
constexpr TValue *to_address(TValue *ptr) noexcept {
    return ptr;
}

template <class TPointer>
constexpr auto to_address(const TPointer &ptr) noexcept {
    return detail::to_address(ptr.operator->());
}

class ICopyConstructible {
   public:
    [[nodiscard]] virtual const type_info *id() const = 0;
    virtual void uninitialized_copy_to(void *dst_ptr) const = 0;
    virtual void uninitialized_move_to(void *dst_ptr) noexcept = 0;
    virtual ~ICopyConstructible() noexcept = default;
};

template <class TAnyIterator, class = iterator_category_t<TAnyIterator>>
class IAnyIteratorAdaptor;

template <class TAnyIterator>
class IAnyIteratorAdaptor<TAnyIterator, std::input_iterator_tag> : public ICopyConstructible {
   public:
    [[nodiscard]] virtual bool operator==(const IAnyIteratorAdaptor &other) const = 0;
    [[nodiscard]] virtual typename TAnyIterator::reference operator*() const = 0;
    [[nodiscard]] virtual typename TAnyIterator::pointer operator->() const = 0;
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
    [[nodiscard]] virtual typename TAnyIterator::difference_type operator-(const IAnyIteratorAdaptor &) const = 0;
    [[nodiscard]] virtual typename TAnyIterator::reference operator[](typename TAnyIterator::difference_type) const = 0;
};

template <class TAnyIterator, class TIterator, class TIteratorTag = iterator_category_t<TAnyIterator>>
class AnyIteratorAdaptor;

template <class TAnyIterator, class TIterator>
class FinalAnyIteratorAdaptor final : public AnyIteratorAdaptor<TAnyIterator, TIterator> {
    using AnyIteratorAdaptor<TAnyIterator, TIterator>::AnyIteratorAdaptor;
};

template <class TAnyIterator, class TIterator>
class AnyIteratorAdaptor<TAnyIterator, TIterator, std::input_iterator_tag> : public IAnyIteratorAdaptor<TAnyIterator> {
    using final_iterator_adaptor_type = FinalAnyIteratorAdaptor<TAnyIterator, TIterator>;

   protected:
    TIterator iterator;

   public:
    inline AnyIteratorAdaptor(const AnyIteratorAdaptor &) = default;
    inline AnyIteratorAdaptor(AnyIteratorAdaptor &&) noexcept = default;

    template <class UIterator, class = std::enable_if_t<std::is_same_v<remove_cvref_t<UIterator>, TIterator>>>
    AnyIteratorAdaptor(UIterator &&iterator) : iterator(static_cast<UIterator &&>(iterator)) {}

    [[nodiscard]] const type_info *id() const override {
        static const type_info unique{};
        return &unique;
    }

    void uninitialized_copy_to(void *dst_ptr) const override {
        ::new (dst_ptr) final_iterator_adaptor_type(static_cast<const final_iterator_adaptor_type &>(*this));
    }

    void uninitialized_move_to(void *dst_ptr) noexcept override {
        ::new (dst_ptr) final_iterator_adaptor_type(static_cast<final_iterator_adaptor_type &&>(*this));
    }

    [[nodiscard]] bool operator==(
        const IAnyIteratorAdaptor<TAnyIterator, std::input_iterator_tag> &other) const override {
        return id() == other.id() and iterator == static_cast<const AnyIteratorAdaptor &>(other).iterator;
    }

    [[nodiscard]] typename TAnyIterator::reference operator*() const override {
        return static_cast<typename TAnyIterator::reference>(*iterator);
    }

    [[nodiscard]] typename TAnyIterator::pointer operator->() const override { return detail::to_address(iterator); }

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
        const IAnyIteratorAdaptor<TAnyIterator> &other) const override {
        return this->id() == other.id() ? this->iterator - static_cast<const AnyIteratorAdaptor &>(other).iterator
                                        : this->iterator - TIterator{};
    }

    [[nodiscard]] typename TAnyIterator::reference operator[](
        typename TAnyIterator::difference_type offset) const override {
        return static_cast<typename TAnyIterator::reference>(this->iterator[offset]);
    }
};

}  // namespace detail

template <class TElement, class TIteratorTag, std::size_t Capacity = default_any_iterator_capacity>
class AnyIterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = std::remove_cv_t<TElement>;
    using pointer = TElement *;
    using reference = TElement &;
    using iterator_category = TIteratorTag;

   private:
    alignas(any_iterator_alignment) std::byte bytes[Capacity];

    using iterator_adaptor_type = detail::IAnyIteratorAdaptor<AnyIterator>;

    inline iterator_adaptor_type &iterator_adaptor() noexcept {
        return *static_cast<iterator_adaptor_type *>(static_cast<void *>(bytes));
    }

    inline const iterator_adaptor_type &iterator_adaptor() const noexcept {
        return *static_cast<const iterator_adaptor_type *>(static_cast<const void *>(bytes));
    }

   public:
    AnyIterator() = delete;

    inline AnyIterator(const AnyIterator &other) { other.iterator_adaptor().uninitialized_copy_to(bytes); }

    inline AnyIterator(AnyIterator &&other) noexcept { other.iterator_adaptor().uninitialized_move_to(bytes); }

   private:
    template <class TIterator, class TEnable = void>
    struct enable_if_iterator_compatible_cvref
        : std::enable_if<detail::iterator_compatible_cvref<TIterator, reference, iterator_category>> {};

    // Don't hide copy and move constructors
    template <class TIterator>
    struct enable_if_iterator_compatible_cvref<
        TIterator,
        std::enable_if_t<std::is_same_v<detail::remove_cvref_t<TIterator>, AnyIterator>>> {};

    // Discard candidate for implicit construction from adaptor type
    template <class TIterator>
    struct enable_if_iterator_compatible_cvref<
        TIterator,
        std::enable_if_t<std::is_same_v<detail::remove_cvref_t<TIterator>, iterator_adaptor_type>>> {};

   public:
    template <class TIterator, class = typename enable_if_iterator_compatible_cvref<TIterator>::type>
    // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
    AnyIterator(TIterator &&iterator) {
        using final_adaptor_type = detail::FinalAnyIteratorAdaptor<AnyIterator, detail::remove_cvref_t<TIterator>>;

        static_assert(sizeof(AnyIterator) >= sizeof(final_adaptor_type));
        static_assert(alignof(AnyIterator) % alignof(final_adaptor_type) == 0);

        ::new (bytes) final_adaptor_type(static_cast<TIterator &&>(iterator));
    }

    inline void swap(AnyIterator &rhs) noexcept {
        AnyIterator tmp(static_cast<AnyIterator &&>(rhs));
        rhs.iterator_adaptor().~iterator_adaptor_type();
        iterator_adaptor().uninitialized_move_to(rhs.bytes);
        iterator_adaptor().~iterator_adaptor_type();
        tmp.iterator_adaptor().uninitialized_move_to(bytes);
    }

    inline AnyIterator &operator=(const AnyIterator &other) {
        if (this != &other) {
            iterator_adaptor().~iterator_adaptor_type();
            other.iterator_adaptor().uninitialized_copy_to(bytes);
        }
        return *this;
    }

    inline AnyIterator &operator=(AnyIterator &&other) noexcept {
        if (this != &other) {
            iterator_adaptor().~iterator_adaptor_type();
            other.iterator_adaptor().uninitialized_move_to(bytes);
        }
        return *this;
    }

    inline ~AnyIterator() noexcept { iterator_adaptor().~iterator_adaptor_type(); }

    // input methods
    static_assert(std::is_base_of_v<std::input_iterator_tag, iterator_category>);

    [[nodiscard]] friend inline bool operator==(const AnyIterator &lhs, const AnyIterator &rhs) {
        return lhs.iterator_adaptor() == rhs.iterator_adaptor();
    }

    [[nodiscard]] friend inline bool operator!=(const AnyIterator &lhs, const AnyIterator &rhs) {
        return not(lhs == rhs);
    }

    [[nodiscard]] inline reference operator*() const { return *iterator_adaptor(); }

    [[nodiscard]] inline pointer operator->() const { return iterator_adaptor().operator->(); }

    inline AnyIterator &operator++() {
        auto &adaptor = iterator_adaptor();
        adaptor.operator++();
        return *this;
    }

    // forward methods
   private:
    struct enable_if_forward
        : std::enable_if<std::is_base_of_v<std::forward_iterator_tag, iterator_category>, enable_if_forward> {};

   public:
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
    AnyIterator &operator--() {
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
    AnyIterator &operator+=(difference_type offset) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        iterator_adaptor() += offset;
        return *this;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    AnyIterator &operator-=(difference_type offset) {
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
    [[nodiscard]] friend AnyIterator operator+(difference_type offset, const AnyIterator &other) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return other.iterator_adaptor() + offset;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] AnyIterator operator-(difference_type offset) const {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return iterator_adaptor() + -offset;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] difference_type operator-(const AnyIterator &other) const {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return iterator_adaptor() - other.iterator_adaptor();
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] reference operator[](difference_type offset) const {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return iterator_adaptor()[offset];
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] friend bool operator<(const AnyIterator &lhs, const AnyIterator &rhs) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return (lhs - rhs) < 0;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] friend bool operator>(const AnyIterator &lhs, const AnyIterator &rhs) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return (lhs - rhs) > 0;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] friend bool operator<=(const AnyIterator &lhs, const AnyIterator &rhs) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return (lhs - rhs) <= 0;
    }

    template <class TEnable = enable_if_random_access, class = typename TEnable::type>
    [[nodiscard]] friend bool operator>=(const AnyIterator &lhs, const AnyIterator &rhs) {
        static_assert(std::is_same_v<TEnable, enable_if_random_access>);
        return (lhs - rhs) >= 0;
    }
};

template <class TElement, class TIteratorTag, std::size_t Capacity>
void swap(
    AnyIterator<TElement, TIteratorTag, Capacity> &lhs, AnyIterator<TElement, TIteratorTag, Capacity> &rhs) noexcept {
    lhs.swap(rhs);
}

namespace detail {

template <class TElement, class TRangeTag, class TIterator, class TEnable = void>
struct AnyRangeTraits {
    using value_type = std::remove_cv_t<TElement>;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
    using reference = TElement &;
    using pointer = TElement *;
    using iterator = TIterator;
};

template <class TElement, class TRangeTag, class TIterator>
struct AnyRangeTraits<
    TElement,
    TRangeTag,
    TIterator,
    std::enable_if_t<std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category_t<TRangeTag>>>>
    : AnyRangeTraits<TElement, forward_range_tag, TIterator> {
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
class FinalAnyRangeAdaptor final : public AnyRangeAdaptor<TAnyRange, TRange> {
    using AnyRangeAdaptor<TAnyRange, TRange>::AnyRangeAdaptor;
};

template <class TAnyRange, class TRange>
class AnyRangeAdaptor<TAnyRange, TRange, input_range_tag> : public IAnyRangeAdaptor<TAnyRange> {
    using final_range_adaptor_type = FinalAnyRangeAdaptor<TAnyRange, TRange>;

   protected:
    TRange range;

   public:
    inline AnyRangeAdaptor(const AnyRangeAdaptor &) = default;
    inline AnyRangeAdaptor(AnyRangeAdaptor &&) noexcept = default;

    template <class URange, class = std::enable_if_t<std::is_same_v<remove_cvref_t<TRange>, remove_cvref_t<URange>>>>
    inline AnyRangeAdaptor(URange &&range) : range(static_cast<URange &&>(range)) {}

    [[nodiscard]] const type_info *id() const override {
        static const type_info unique{};
        return &unique;
    }

    void uninitialized_copy_to(void *dst_ptr) const override {
        ::new (dst_ptr) final_range_adaptor_type(static_cast<const final_range_adaptor_type &>(*this));
    }

    void uninitialized_move_to(void *dst_ptr) noexcept override {
        ::new (dst_ptr) final_range_adaptor_type(static_cast<final_range_adaptor_type &&>(*this));
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

template <
    class TElement,
    class TRangeTag,
    class TAnyIterator = AnyIterator<TElement, detail::iterator_category_t<TRangeTag>>,
    std::size_t Capacity = default_any_range_capacity>
class BasicAnyRange : public detail::AnyRangeTraits<TElement, TRangeTag, TAnyIterator> {
   public:
    using range_category = TRangeTag;

   private:
    alignas(any_range_alignment) mutable std::byte bytes[Capacity];

    using iterator_category = detail::iterator_category_t<range_category>;
    using range_adaptor_type = detail::IAnyRangeAdaptor<BasicAnyRange, range_category>;

    inline range_adaptor_type &range_adaptor() const noexcept {
        return *static_cast<range_adaptor_type *>(static_cast<void *>(bytes));
    }

    using traits = detail::AnyRangeTraits<TElement, TRangeTag, TAnyIterator>;

   public:
    using typename traits::difference_type;
    using typename traits::iterator;
    using typename traits::pointer;
    using typename traits::reference;
    using typename traits::size_type;
    using typename traits::value_type;

    BasicAnyRange() = delete;

    inline BasicAnyRange(const BasicAnyRange &other) { other.range_adaptor().uninitialized_copy_to(bytes); }

    inline BasicAnyRange(BasicAnyRange &&other) noexcept { other.range_adaptor().uninitialized_move_to(bytes); }

   private:
    template <class TRange, class TEnable = void>
    struct enable_if_range_compatible : std::enable_if<detail::range_compatible<TRange, reference, range_category>> {};

    // Don't hide copy and move constructors
    template <class TRange>
    struct enable_if_range_compatible<
        TRange,
        std::enable_if_t<std::is_same_v<detail::remove_cvref_t<TRange>, BasicAnyRange>>> {};

   public:
    template <class TRange, class = typename enable_if_range_compatible<TRange>::type>
    // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
    BasicAnyRange(TRange &&range) {
        using final_adaptor_type = detail::FinalAnyRangeAdaptor<BasicAnyRange, TRange>;

        static_assert(sizeof(BasicAnyRange) >= sizeof(final_adaptor_type));
        static_assert(alignof(BasicAnyRange) % alignof(final_adaptor_type) == 0);

        ::new (static_cast<void *>(bytes)) final_adaptor_type(static_cast<TRange &&>(range));
    }

    inline void swap(BasicAnyRange &rhs) noexcept {
        BasicAnyRange tmp(static_cast<BasicAnyRange &&>(rhs));
        rhs.range_adaptor().~range_adaptor_type();
        range_adaptor().uninitialized_move_to(rhs.bytes);
        range_adaptor().~range_adaptor_type();
        tmp.range_adaptor().uninitialized_move_to(bytes);
    }

    inline BasicAnyRange &operator=(const BasicAnyRange &other) {
        if (this != &other) {
            range_adaptor().~range_adaptor_type();
            other.range_adaptor().uninitialized_copy_to(bytes);
        }
        return *this;
    }

    inline BasicAnyRange &operator=(BasicAnyRange &&other) noexcept {
        if (this != &other) {
            range_adaptor().~range_adaptor_type();
            other.range_adaptor().uninitialized_move_to(bytes);
        }
        return *this;
    }

    inline ~BasicAnyRange() noexcept { range_adaptor().~range_adaptor_type(); }

    // input methods
    static_assert(std::is_base_of_v<input_range_tag, range_category>);

    [[nodiscard]] inline iterator begin() const { return range_adaptor().begin(); }

    [[nodiscard]] inline iterator end() const { return range_adaptor().end(); }

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

template <class TElement, class TRangeTag, class TAnyIterator, std::size_t Capacity>
void swap(
    BasicAnyRange<TElement, TRangeTag, TAnyIterator, Capacity> &lhs,
    BasicAnyRange<TElement, TRangeTag, TAnyIterator, Capacity> &rhs) noexcept {
    lhs.swap(rhs);
}

template <
    class TElement,
    class TRangeTag,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyRange = BasicAnyRange<
    TElement,
    TRangeTag,
    AnyIterator<TElement, detail::iterator_category_t<TRangeTag>, IteratorCapacity>,
    RangeCapacity>;

template <
    class TElement,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyInputRange = AnyRange<TElement, input_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TElement,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyForwardRange = AnyRange<TElement, forward_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TElement,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyBidirectionalRange = AnyRange<TElement, bidirectional_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TElement,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnyRandomAccessRange = AnyRange<TElement, random_access_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TElement,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnySizedInputRange = AnyRange<TElement, sized_input_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TElement,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnySizedForwardRange = AnyRange<TElement, sized_forward_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TElement,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnySizedBidirectionalRange = AnyRange<TElement, sized_bidirectional_range_tag, RangeCapacity, IteratorCapacity>;

template <
    class TElement,
    std::size_t RangeCapacity = default_any_range_capacity,
    std::size_t IteratorCapacity = default_any_iterator_capacity>
using AnySizedRandomAccessRange = AnyRange<TElement, sized_random_access_range_tag, RangeCapacity, IteratorCapacity>;

template <class TElement, class TRangeTag, class... TRanges>
using AnyRangeFor = AnyRange<
    // cause substitution failure if any type in pack does not satisfy alignment requirement for range or iterator
    std::enable_if_t<(... and detail::is_aligned_to_v<TRanges, any_range_alignment>), TElement>,
    std::enable_if_t<(... and detail::is_aligned_to_v<detail::iterator_t<TRanges>, any_iterator_alignment>), TRangeTag>,
    // add sizeof vtable pointer
    sizeof(void *) + std::max({sizeof(TRanges)...}),
    sizeof(void *) + std::max({sizeof(detail::iterator_t<TRanges>)...})>;

template <class TElement, class... TRanges>
using AnyInputRangeFor = AnyRangeFor<TElement, input_range_tag, TRanges...>;

template <class TElement, class... TRanges>
using AnyForwardRangeFor = AnyRangeFor<TElement, forward_range_tag, TRanges...>;

template <class TElement, class... TRanges>
using AnyBidirectionalRangeFor = AnyRangeFor<TElement, bidirectional_range_tag, TRanges...>;

template <class TElement, class... TRanges>
using AnyRandomAccessRangeFor = AnyRangeFor<TElement, random_access_range_tag, TRanges...>;

template <class TElement, class... TRanges>
using AnySizedInputRangeFor = AnyRangeFor<TElement, sized_input_range_tag, TRanges...>;

template <class TElement, class... TRanges>
using AnySizedForwardRangeFor = AnyRangeFor<TElement, sized_forward_range_tag, TRanges...>;

template <class TElement, class... TRanges>
using AnySizedBidirectionalRangeFor = AnyRangeFor<TElement, sized_bidirectional_range_tag, TRanges...>;

template <class TElement, class... TRanges>
using AnySizedRandomAccessRangeFor = AnyRangeFor<TElement, sized_random_access_range_tag, TRanges...>;

}  // namespace tt::stl
