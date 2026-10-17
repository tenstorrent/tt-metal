// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

// Positional access to the graph_kernel bindings. The program factory binds input I as tensor "in<I>"
// and dataflow buffer "in<I>", and intermediate K as dataflow buffer "t<K>"; the generated bindings
// header only offers named lookup, so this spells the name from the index at compile time and
// resolves it through get_token_if_present. An index with no binding dereferences a null constexpr
// pointer, which is a compile error.
namespace graph_kernel {

template <size_t I, char C0, char C1 = '\0'>
struct IndexedName {
    static constexpr size_t prefix = (C1 == '\0') ? 1 : 2;
    static constexpr size_t digits = I < 10 ? 1 : (I < 100 ? 2 : 3);
    // Exactly strlen + 1: TemplateString compares lengths, so the array must match the literal the
    // generated get_token_if_present() was built from.
    static constexpr size_t size = prefix + digits + 1;
    struct Chars {
        char v[size];
    };
    static constexpr Chars make() {
        Chars c{};
        c.v[0] = C0;
        if constexpr (prefix == 2) {
            c.v[1] = C1;
        }
        size_t n = I;
        for (size_t k = digits; k > 0; --k) {
            c.v[prefix - 1 + k] = static_cast<char>('0' + n % 10);
            n /= 10;
        }
        c.v[size - 1] = '\0';
        return c;
    }
    static constexpr Chars chars = make();
};

template <size_t I>
using InputName = IndexedName<I, 'i', 'n'>;
template <size_t I>
using IntermName = IndexedName<I, 't'>;

template <size_t I>
constexpr auto input_dfb() {
    return *dfb::get_token_if_present<InputName<I>::chars.v>();
}

template <size_t I>
constexpr auto input_tensor() {
    return *tensor::get_token_if_present<InputName<I>::chars.v>();
}

template <size_t I>
constexpr auto interm_dfb() {
    return *dfb::get_token_if_present<IntermName<I>::chars.v>();
}

}  // namespace graph_kernel
