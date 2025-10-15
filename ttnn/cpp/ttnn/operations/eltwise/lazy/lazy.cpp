// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/lazy/lazy.hpp"

#include <numbers>

namespace ttnn::operations::lazy {

Function UnaryFn::operator()(ExpressionView first) const { return lazy::defer(operation, first).value(); }

Function UnaryWithParamFn::operator()(ExpressionView first, Param second) const {
    return lazy::defer(operation, first, second).value();
}

Function RUnaryWithParamFn::operator()(Param first, ExpressionView second) const {
    return lazy::defer(operation, second, first).value();
}

Function BinaryFn::operator()(ExpressionView first, ExpressionView second) const {
    return lazy::defer(operation, first, second).value();
}

Function RBinaryFn::operator()(ExpressionView first, ExpressionView second) const {
    return lazy::defer(operation, second, first).value();
}

Function TernaryFn::operator()(ExpressionView first, ExpressionView second, ExpressionView third) const {
    return lazy::defer(operation, first, second, third).value();
}

Function DivFn::operator()(Param first, ExpressionView second) const { return lazy::mul(first, lazy::recip(second)); }

Function RDivFn::operator()(ExpressionView first, Param second) const { return lazy::div(second, first); }

template <typename First, typename Second>
auto _compare(UnaryFn compare_to_zero, First first, Second second) {
    return compare_to_zero(lazy::sub(first, second));
}

Function CompareFn::operator()(ExpressionView first, ExpressionView second) const {
    return lazy::_compare(operation, first, second);
}

Function CompareFn::operator()(ExpressionView first, Param second) const {
    return lazy::_compare(operation, first, second);
}

Function CompareFn::operator()(Param first, ExpressionView second) const {
    return lazy::_compare(operation, first, second);
}

inline auto _nez(ExpressionView first) { return lazy::nez(first); }

constexpr auto _nez(Param first) {
    return std::visit([](auto value) { return value != 0; }, first);
}

template <typename First, typename Second>
auto _logical_binary(OverloadedBinaryFn operation, First first, Second second) {
    return lazy::nez(operation(lazy::_nez(first), lazy::_nez(second)));
}

Function LogicalBinaryFn::operator()(ExpressionView first, ExpressionView second) const {
    return lazy::_logical_binary(operation, first, second);
}

Function LogicalBinaryFn::operator()(ExpressionView first, Param second) const {
    return lazy::_logical_binary(operation, first, second);
}

Function LogicalBinaryFn::operator()(Param first, ExpressionView second) const {
    return lazy::_logical_binary(operation, first, second);
}

template <typename Input, typename Other>
auto _where(ExpressionView condition, Input input, Other other) {
    return lazy::add(lazy::mul(lazy::nez(condition), input), lazy::mul(lazy::eqz(condition), other));
}

Function WhereFn::operator()(ExpressionView condition, ExpressionView input, Param other) const {
    return lazy::_where(condition, input, other);
}

Function WhereFn::operator()(ExpressionView condition, Param input, ExpressionView other) const {
    return lazy::_where(condition, input, other);
}

Function WhereFn::operator()(ExpressionView condition, Param input, Param other) const {
    return lazy::_where(condition, input, other);
}

inline auto _ltz(ExpressionView first) { return lazy::ltz(first); }

constexpr auto _ltz(Param first) {
    return std::visit([](auto value) { return value < 0; }, first);
}

inline auto _gtz(ExpressionView first) { return lazy::gtz(first); }

constexpr auto _gtz(Param first) {
    return std::visit([](auto value) { return value > 0; }, first);
}

inline auto _gez(ExpressionView first) { return lazy::gez(first); }

constexpr auto _gez(Param first) {
    return std::visit([](auto value) { return value >= 0; }, first);
}

template <typename Input, typename Other>
constexpr auto _where(bool condition, Input input, Other other) {
    return condition ? input : other;
}

template <typename Condition, typename Input, typename... Other>
    requires(sizeof...(Other) > 1)
auto _where(Condition condition, Input input, Other... other) {
    return lazy::_where(condition, input, lazy::_where(other...));
}

template <typename First, typename Second>
auto _atan2(First y, Second x) {
    constexpr auto pi = std::numbers::pi_v<float>;
    const auto atan = lazy::atan(lazy::div(y, x));
    return lazy::_where(
        lazy::_gtz(x),
        atan,
        lazy::logical_and(lazy::_ltz(x), lazy::_gez(y)),
        lazy::add(atan, pi),
        lazy::logical_and(lazy::_ltz(x), lazy::_ltz(y)),
        lazy::sub(atan, pi),
        lazy::_gtz(y),
        pi * 2,
        lazy::_ltz(y),
        -pi * 2,
        0);
}

Function Atan2Fn::operator()(ExpressionView first, Param second) const { return lazy::_atan2(first, second); }

Function Atan2Fn::operator()(Param first, ExpressionView second) const { return lazy::_atan2(first, second); }

Function Atan2Fn::operator()(ExpressionView first, ExpressionView second) const { return lazy::_atan2(first, second); }

}  // namespace ttnn::operations::lazy
