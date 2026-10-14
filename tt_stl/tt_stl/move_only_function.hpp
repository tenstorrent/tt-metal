// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Written for tt-metal (GitHub #57444), adapting the designs of two projects; see NOTICE:
//   - zoo (https://github.com/thecppzoo/zoo), MIT License, Copyright (c) 2017 thecppzoo: the layout
//     of zoo::VTableFunction -- a fixed inline buffer plus one pointer to a per-type table holding
//     the invoker -- and an empty state whose table entry throws.
//   - LLVM (llvm/ADT/FunctionExtras.h, llvm::unique_function), Apache License v2.0 with LLVM
//     Exceptions, https://llvm.org/LICENSE.txt: the trivial-callable fast path, where a target that
//     is trivially movable and destructible is moved by copying bytes and destroyed by doing
//     nothing, flagged with the callback pointer rather than looked up.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <utility>

namespace ttsl {

namespace detail {

// With the table pointer, sizeof 32, as std::function on libstdc++.
inline constexpr std::size_t kMoveOnlyFunctionInlineSize = 3 * sizeof(void*);
inline constexpr std::size_t kMoveOnlyFunctionInlineAlign = alignof(void*);

template <typename T>
struct is_in_place_type : std::false_type {};
template <typename T>
struct is_in_place_type<std::in_place_type_t<T>> : std::true_type {};

template <typename T>
struct is_std_function : std::false_type {};
template <typename R, typename... A>
struct is_std_function<std::function<R(A...)>> : std::true_type {};

template <typename T>
inline constexpr bool is_nullable_callable_v =
    std::is_pointer_v<T> || std::is_member_pointer_v<T> || is_std_function<T>::value;

template <typename F, typename Self, typename R, typename... Args>
concept MoveOnlyFunctionTarget =
    !std::is_same_v<std::remove_cvref_t<F>, Self> && !is_in_place_type<std::remove_cvref_t<F>>::value &&
    std::is_invocable_r_v<R, std::decay_t<F>, Args...> && std::is_invocable_r_v<R, std::decay_t<F>&, Args...>;

template <typename T>
    requires is_nullable_callable_v<std::decay_t<T>>
bool is_empty_callable(const T& f) noexcept {
    if constexpr (is_std_function<T>::value) {
        return !static_cast<bool>(f);
    } else {
        return f == nullptr;
    }
}

// A target lives in the buffer when it fits and its move cannot throw: moves happen inside this
// type's noexcept move, so a throwing one has to go to the heap.
template <typename T>
inline constexpr bool stored_inline_v =
    sizeof(T) <= kMoveOnlyFunctionInlineSize && alignof(T) <= kMoveOnlyFunctionInlineAlign &&
    std::is_nothrow_move_constructible_v<T>;

// Moved by copying bytes and never destroyed: the llvm::unique_function criterion.
template <typename T>
inline constexpr bool trivial_inline_v =
    stored_inline_v<T> && std::is_trivially_move_constructible_v<T> && std::is_trivially_destructible_v<T>;

template <typename R, typename F, typename... A>
R invoke_r(F& f, A&&... args) {
    if constexpr (std::is_void_v<R>) {
        std::invoke(f, std::forward<A>(args)...);
    } else {
        return std::invoke(f, std::forward<A>(args)...);
    }
}

}  // namespace detail

// A polyfill for C++23 std::move_only_function: a type-erased wrapper for any callable that is
// move-constructible, including ones that cannot be copied, such as a lambda capturing a
// std::unique_ptr.
//
// It follows std::move_only_function except that:
//   - calling an empty instance throws std::bad_function_call, as std::function does, rather than
//     being undefined;
//   - only the unqualified R(Args...) signature is provided, not the const, reference or noexcept
//     qualified forms.
//
// Captures up to 24 bytes whose move cannot throw are stored inline; anything else is heap
// allocated. A moved-from instance is empty.
//
// TODO(#57444): replace with std::move_only_function once every supported standard library ships
// it; libc++ does not as of LLVM 20. Calling an empty instance then becomes undefined.
template <typename Signature>
class move_only_function;

template <typename R, typename... Args>
class move_only_function<R(Args...)> {
    // One table per stored type. Only call is used for every target; relocate and destroy are
    // skipped when the tag bits below say so.
    struct VTable {
        R (*call)(void* storage, Args&&... args);
        void (*relocate)(void* dst, void* src) noexcept;  // move-construct into dst, then destroy src
        void (*destroy)(void* storage) noexcept;
    };

    // The table pointer's low bits record what can be done without it:
    //   kBytewiseRelocate  moving is a memcpy of the buffer -- trivial inline targets
    //   kNoDestroy         destroying does nothing          -- trivially destructible inline targets
    //   kHeapPointer       moving copies one pointer        -- every heap target
    // A heap target is moved by copying only its pointer, not the whole buffer: the pointer was just
    // written as one 8-byte store, and reading it back as 24 bytes defeats store-to-load forwarding,
    // which measured at about 10 ns per move.
    static constexpr std::uintptr_t kBytewiseRelocate = 1;
    static constexpr std::uintptr_t kNoDestroy = 2;
    static constexpr std::uintptr_t kHeapPointer = 4;
    static constexpr std::uintptr_t kTagMask = kBytewiseRelocate | kNoDestroy | kHeapPointer;
    static_assert(alignof(VTable) > kTagMask, "the tag bits need zero bits in the table address");

public:
    using result_type = R;

    move_only_function() noexcept : tag_(empty_tag()) {}
    move_only_function(std::nullptr_t) noexcept : tag_(empty_tag()) {}

    move_only_function(move_only_function&& other) noexcept { take(other); }

    template <typename F>
        requires(
            detail::MoveOnlyFunctionTarget<F, move_only_function, R, Args...> &&
            !detail::is_nullable_callable_v<std::decay_t<F>>)
    move_only_function(F&& f) {
        emplace<std::decay_t<F>>(std::forward<F>(f));
    }

    template <typename F>
        requires(
            detail::MoveOnlyFunctionTarget<F, move_only_function, R, Args...> &&
            detail::is_nullable_callable_v<std::decay_t<F>>)
    move_only_function(F&& f) : tag_(empty_tag()) {
        if (!detail::is_empty_callable(f)) {
            emplace<std::decay_t<F>>(std::forward<F>(f));
        }
    }

    template <typename T, typename... CArgs>
        requires(std::is_constructible_v<T, CArgs...> && std::is_invocable_r_v<R, T&, Args...>)
    explicit move_only_function(std::in_place_type_t<T>, CArgs&&... args) {
        emplace<T>(std::forward<CArgs>(args)...);
    }

    template <typename T, typename U, typename... CArgs>
        requires(
            std::is_constructible_v<T, std::initializer_list<U>&, CArgs...> && std::is_invocable_r_v<R, T&, Args...>)
    explicit move_only_function(std::in_place_type_t<T>, std::initializer_list<U> il, CArgs&&... args) {
        emplace<T>(il, std::forward<CArgs>(args)...);
    }

    move_only_function(const move_only_function&) = delete;
    move_only_function& operator=(const move_only_function&) = delete;

    move_only_function& operator=(move_only_function&& other) noexcept {
        if (this != &other) {
            destroy_target();
            take(other);
        }
        return *this;
    }

    move_only_function& operator=(std::nullptr_t) noexcept {
        destroy_target();
        tag_ = empty_tag();
        return *this;
    }

    template <typename F>
        requires std::is_constructible_v<move_only_function, F>
    move_only_function& operator=(F&& f) {
        move_only_function(std::forward<F>(f)).swap(*this);
        return *this;
    }

    ~move_only_function() { destroy_target(); }

    explicit operator bool() const noexcept { return table() != &kEmptyTable; }

    // Calling an empty instance throws through the empty table, with no branch here.
    R operator()(Args... args) { return table()->call(buffer_, std::forward<Args>(args)...); }

    void swap(move_only_function& other) noexcept {
        if (this != &other) {
            move_only_function tmp(std::move(other));
            other = std::move(*this);
            *this = std::move(tmp);
        }
    }

    friend void swap(move_only_function& a, move_only_function& b) noexcept { a.swap(b); }

    friend bool operator==(const move_only_function& f, std::nullptr_t) noexcept { return !f; }

private:
    [[noreturn]] static R call_empty(void*, Args&&...) { throw std::bad_function_call(); }

    template <typename T>
    static T& inline_target(void* storage) noexcept {
        return *std::launder(static_cast<T*>(storage));
    }

    template <typename T>
    static T*& heap_target(void* storage) noexcept {
        return *std::launder(static_cast<T**>(storage));
    }

    template <typename T>
    static R call_inline(void* storage, Args&&... args) {
        return detail::invoke_r<R>(inline_target<T>(storage), std::forward<Args>(args)...);
    }

    template <typename T>
    static void relocate_inline(void* dst, void* src) noexcept {
        T& from = inline_target<T>(src);
        ::new (dst) T(std::move(from));
        from.~T();  // NOLINT(bugprone-use-after-move): ending the moved-from source's lifetime is the point
    }

    template <typename T>
    static void destroy_inline(void* storage) noexcept {
        inline_target<T>(storage).~T();
    }

    template <typename T>
    static R call_heap(void* storage, Args&&... args) {
        return detail::invoke_r<R>(*heap_target<T>(storage), std::forward<Args>(args)...);
    }

    template <typename T>
    static void destroy_heap(void* storage) noexcept {
        delete heap_target<T>(storage);
    }

    static constexpr VTable kEmptyTable{&call_empty, nullptr, nullptr};

    template <typename T>
    static constexpr VTable kInlineTable{&call_inline<T>, &relocate_inline<T>, &destroy_inline<T>};

    // A heap target moves by copying its pointer, so it needs no relocate entry.
    template <typename T>
    static constexpr VTable kHeapTable{&call_heap<T>, nullptr, &destroy_heap<T>};

    static std::uintptr_t make_tag(const VTable* table, std::uintptr_t bits) noexcept {
        return reinterpret_cast<std::uintptr_t>(table) | bits;
    }

    static std::uintptr_t empty_tag() noexcept { return make_tag(&kEmptyTable, kBytewiseRelocate | kNoDestroy); }

    const VTable* table() const noexcept { return reinterpret_cast<const VTable*>(tag_ & ~kTagMask); }

    template <typename T, typename... A>
    void emplace(A&&... args) {
        if constexpr (detail::stored_inline_v<T>) {
            ::new (static_cast<void*>(buffer_)) T(std::forward<A>(args)...);
            constexpr std::uintptr_t bits = (detail::trivial_inline_v<T> ? kBytewiseRelocate : 0) |
                                            (std::is_trivially_destructible_v<T> ? kNoDestroy : 0);
            tag_ = make_tag(&kInlineTable<T>, bits);
        } else {
            T* target = new T(std::forward<A>(args)...);
            ::new (static_cast<void*>(buffer_)) T*(target);
            tag_ = make_tag(&kHeapTable<T>, kHeapPointer);
        }
    }

    // Moves other's target into this, which holds none, and leaves other empty.
    //
    // A bytewise move copies the whole buffer, including bytes a small target (or the empty state)
    // never wrote. Copying indeterminate bytes as unsigned storage is well-defined, but gcc reports
    // it as -W(maybe-)uninitialized once inlined. Zeroing those bytes instead adds a store to every
    // construction and was not faster. __builtin_memcpy keeps the copy out of glibc's fortified memcpy
    // wrapper, so the suppression below applies to it.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
    void take(move_only_function& other) noexcept {
        tag_ = other.tag_;
        if (tag_ & kBytewiseRelocate) {
            __builtin_memcpy(buffer_, other.buffer_, detail::kMoveOnlyFunctionInlineSize);
        } else if (tag_ & kHeapPointer) {
            __builtin_memcpy(buffer_, other.buffer_, sizeof(void*));
        } else {
            table()->relocate(buffer_, other.buffer_);
        }
        other.tag_ = empty_tag();
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

    void destroy_target() noexcept {
        if (!(tag_ & kNoDestroy)) {
            table()->destroy(buffer_);
        }
    }

    alignas(detail::kMoveOnlyFunctionInlineAlign) std::byte buffer_[detail::kMoveOnlyFunctionInlineSize];
    std::uintptr_t tag_;
};

}  // namespace ttsl
