// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: Portions copyright LLVM Project (https://llvm.org): llvm::unique_function
// SPDX-FileCopyrightText: Portions copyright (c) 2017 thecppzoo (https://github.com/thecppzoo/zoo): zoo::VTableFunction
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <utility>

namespace ttsl {

template <typename Signature>
class move_only_function;

namespace detail {

// With the table pointer, four pointers: the size of std::function on libstdc++.
inline constexpr std::size_t kMoveOnlyFunctionInlineSize = 3 * sizeof(void*);
// At least 8, so 8-byte-aligned captures stay inline on 32-bit targets too.
inline constexpr std::size_t kMoveOnlyFunctionInlineAlign = alignof(void*) > alignof(std::uint64_t)
                                                                ? alignof(void*)
                                                                : alignof(std::uint64_t);

template <typename T>
struct is_in_place_type : std::false_type {};
template <typename T>
struct is_in_place_type<std::in_place_type_t<T>> : std::true_type {};

// How the table's call entry takes each argument: scalars by value, so they can travel in registers,
// anything else by reference, so a class-type argument is not moved again. libstdc++ does the same.
template <typename T>
using param_t = std::conditional_t<std::is_scalar_v<T>, T, T&&>;

template <typename T>
struct is_move_only_function : std::false_type {};
template <typename Signature>
struct is_move_only_function<move_only_function<Signature>> : std::true_type {};

template <typename T>
inline constexpr bool is_nullable_callable_v =
    std::is_pointer_v<T> || std::is_member_pointer_v<T> || is_move_only_function<T>::value;

template <typename Result, typename Value>
inline constexpr bool binds_to_temporary_v =
#if __has_builtin(__reference_converts_from_temporary)
    __reference_converts_from_temporary(Result, Value);
#else
    false;
#endif

template <typename Target, typename Result, typename... Args>
concept IsCallableFrom =
    std::is_invocable_r_v<Result, Target, Args...> && std::is_invocable_r_v<Result, Target&, Args...> &&
    !binds_to_temporary_v<Result, std::invoke_result_t<Target, Args...>> &&
    !binds_to_temporary_v<Result, std::invoke_result_t<Target&, Args...>>;

template <typename Callable, typename Self, typename Result, typename... Args>
concept MoveOnlyFunctionTarget =
    !std::is_same_v<std::remove_cvref_t<Callable>, Self> && !is_in_place_type<std::remove_cvref_t<Callable>>::value &&
    IsCallableFrom<std::decay_t<Callable>, Result, Args...>;

template <typename T>
inline constexpr bool stored_inline_v =
    sizeof(T) <= kMoveOnlyFunctionInlineSize && alignof(T) <= kMoveOnlyFunctionInlineAlign &&
    std::is_nothrow_move_constructible_v<T>;

// Moved by copying bytes and never destroyed: the llvm::unique_function criterion.
template <typename T>
inline constexpr bool trivial_inline_v =
    stored_inline_v<T> && std::is_trivially_move_constructible_v<T> && std::is_trivially_destructible_v<T>;

template <typename Result, typename Callable, typename... Args>
Result invoke_r(Callable& callable, Args&&... args) {
    if constexpr (std::is_void_v<Result>) {
        std::invoke(callable, std::forward<Args>(args)...);
    } else {
        return std::invoke(callable, std::forward<Args>(args)...);
    }
}

}  // namespace detail

// A C++20 implementation of C++23's std::move_only_function, to use until the standard library has
// it: a type-erased wrapper for any callable that is move-constructible, including ones that cannot
// be copied, such as a lambda capturing a std::unique_ptr.
//
// It follows std::move_only_function except that:
//   - calling an empty instance throws std::bad_function_call, as std::function does, rather than
//     being undefined;
//   - only the unqualified R(Args...) signature is provided, not the const, reference or noexcept
//     qualified forms.
template <typename Signature>
class move_only_function;

template <typename R, typename... Args>
class move_only_function<R(Args...)> {
    struct alignas(8) VTable {  // room for the three tag bits on 32-bit targets too
        R (*call)(void* storage, detail::param_t<Args>... args);
        void (*relocate)(void* dst, void* src) noexcept;  // move-construct into dst, then destroy src
        void (*destroy)(void* storage) noexcept;
    };

    // The table pointer's low bits record what can be done without the table.
    static constexpr std::uintptr_t kBytewiseRelocate = 1;  // memcpy move: trivial inline targets
    static constexpr std::uintptr_t kNoDestroy = 2;         // no-op destroy: trivially destructible inline targets
    static constexpr std::uintptr_t kHeapPointer = 4;       // pointer-copy move: every heap target
    static constexpr std::uintptr_t kTagMask = kBytewiseRelocate | kNoDestroy | kHeapPointer;
    static_assert(alignof(VTable) > kTagMask, "the tag bits need zero bits in the table address");

public:
    using result_type = R;

    move_only_function() noexcept : tag_(empty_tag()) {}
    move_only_function(std::nullptr_t) noexcept : tag_(empty_tag()) {}

    move_only_function(move_only_function&& other) noexcept { take(std::move(other)); }

    template <typename F>
        requires detail::MoveOnlyFunctionTarget<F, move_only_function, R, Args...>
    move_only_function(F&& f) {
        if constexpr (detail::is_nullable_callable_v<std::remove_cvref_t<F>>) {
            if (f == nullptr) {
                tag_ = empty_tag();
                return;
            }
        }
        emplace<std::decay_t<F>>(std::forward<F>(f));
    }

    template <typename T, typename... CArgs>
        requires(std::is_constructible_v<T, CArgs...> && detail::IsCallableFrom<T, R, Args...>)
    explicit move_only_function(std::in_place_type_t<T>, CArgs&&... args) {
        emplace<T>(std::forward<CArgs>(args)...);
    }

    template <typename T, typename U, typename... CArgs>
        requires(
            std::is_constructible_v<T, std::initializer_list<U>&, CArgs...> && detail::IsCallableFrom<T, R, Args...>)
    explicit move_only_function(std::in_place_type_t<T>, std::initializer_list<U> il, CArgs&&... args) {
        emplace<T>(il, std::forward<CArgs>(args)...);
    }

    move_only_function& operator=(move_only_function&& other) noexcept {
        if (this != &other) {
            replace(std::move(other));
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

    explicit operator bool() const noexcept {
        return table()->destroy != nullptr;  // only kEmptyTable lacks one; its address differs per shared library
    }

    // Calling an empty instance throws through the empty table, with no branch here.
    // NOLINTNEXTLINE(performance-unnecessary-value-param): std::move_only_function's signature; args are forwarded
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
    [[noreturn]] static R call_empty(void*, detail::param_t<Args>...) { throw std::bad_function_call(); }

    template <typename T>
    static T& inline_target(void* storage) noexcept {
        return *std::launder(static_cast<T*>(storage));
    }

    template <typename T>
    static T& heap_target(void* storage) noexcept {
        return **std::launder(static_cast<T**>(storage));
    }

    template <typename T>
    static R call_inline(void* storage, detail::param_t<Args>... args) {
        return detail::invoke_r<R>(inline_target<T>(storage), std::forward<detail::param_t<Args>>(args)...);
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
    static R call_heap(void* storage, detail::param_t<Args>... args) {
        return detail::invoke_r<R>(heap_target<T>(storage), std::forward<detail::param_t<Args>>(args)...);
    }

    template <typename T>
    static void destroy_heap(void* storage) noexcept {
        delete *std::launder(static_cast<T**>(storage));
    }

    static constexpr VTable kEmptyTable{&call_empty, nullptr, nullptr};

    template <typename T>
    static constexpr VTable kInlineTable{&call_inline<T>, &relocate_inline<T>, &destroy_inline<T>};

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

    // A bytewise move copies the whole buffer, including bytes a small target (or the empty state)
    // never wrote. Copying indeterminate bytes as unsigned storage is well-defined, but gcc reports
    // it as -W(maybe-)uninitialized once inlined. Zeroing those bytes instead adds a store to every
    // construction and was not faster. __builtin_memcpy keeps the copy out of glibc's fortified memcpy
    // wrapper, so the suppression below applies to it. It covers replace() too, where gcc cannot see that a
    // heap tag means the pointer was written.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
    void take(move_only_function&& other) noexcept {
        tag_ = other.tag_;
        if (tag_ & kBytewiseRelocate) {
            __builtin_memcpy(buffer_, other.buffer_, detail::kMoveOnlyFunctionInlineSize);
        } else if (tag_ & kHeapPointer) {
            // Only the pointer: it was just written as one 8-byte store, and reading it back as 24 bytes
            // defeats store-to-load forwarding, which measured at about 10 ns per move.
            __builtin_memcpy(buffer_, other.buffer_, sizeof(void*));
        } else {
            table()->relocate(buffer_, other.buffer_);
        }
        other.tag_ = empty_tag();
    }

    // Takes other's target in place of the current one. other may be owned by the current target, as when
    // a callback replaces itself with its successor, so it is read before that target is destroyed.
    void replace(move_only_function&& other) noexcept {
        if (tag_ & kNoDestroy) {
            take(std::move(other));
        } else if (tag_ & kHeapPointer) {
            alignas(void*) std::byte old_target[sizeof(void*)];
            __builtin_memcpy(old_target, buffer_, sizeof(void*));
            const VTable* old_table = table();
            take(std::move(other));
            old_table->destroy(old_target);
        } else {
            move_only_function incoming(std::move(other));
            destroy_target();
            take(std::move(incoming));
        }
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
