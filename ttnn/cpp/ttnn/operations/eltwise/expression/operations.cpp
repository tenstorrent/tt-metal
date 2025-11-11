// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/expression/operations.hpp"

#include <numbers>

namespace ttnn::operations::expression {

// Unary
Function operator-(ExpressionView first) { return expression::neg(first); }

// Binary
Function operator+(ExpressionView first, ExpressionView second) { return expression::add(first, second); }
Function operator+(ExpressionView first, const Tensor& second) { return expression::add(first, second); }
Function operator+(ExpressionView first, Param second) { return expression::add(first, second); }
Function operator+(const Tensor& first, ExpressionView second) { return expression::add(first, second); }
Function operator+(Param first, ExpressionView second) { return expression::add(first, second); }

Function operator-(ExpressionView first, ExpressionView second) { return expression::sub(first, second); }
Function operator-(ExpressionView first, const Tensor& second) { return expression::sub(first, second); }
Function operator-(ExpressionView first, Param second) { return expression::sub(first, second); }
Function operator-(const Tensor& first, ExpressionView second) { return expression::sub(first, second); }
Function operator-(Param first, ExpressionView second) { return expression::sub(first, second); }

Function operator*(ExpressionView first, ExpressionView second) { return expression::mul(first, second); }
Function operator*(ExpressionView first, const Tensor& second) { return expression::mul(first, second); }
Function operator*(ExpressionView first, Param second) { return expression::mul(first, second); }
Function operator*(const Tensor& first, ExpressionView second) { return expression::mul(first, second); }
Function operator*(Param first, ExpressionView second) { return expression::mul(first, second); }

Function operator/(ExpressionView first, ExpressionView second) { return expression::div(first, second); }
Function operator/(ExpressionView first, const Tensor& second) { return expression::div(first, second); }
Function operator/(ExpressionView first, Param second) { return expression::div(first, second); }
Function operator/(const Tensor& first, ExpressionView second) { return expression::div(first, second); }
Function operator/(Param first, ExpressionView second) { return expression::div(first, second); }

Function operator<(ExpressionView first, ExpressionView second) { return expression::lt(first, second); }
Function operator<(ExpressionView first, const Tensor& second) { return expression::lt(first, second); }
Function operator<(ExpressionView first, Param second) { return expression::lt(first, second); }
Function operator<(const Tensor& first, ExpressionView second) { return expression::lt(first, second); }
Function operator<(Param first, ExpressionView second) { return expression::lt(first, second); }

Function operator>(ExpressionView first, ExpressionView second) { return expression::gt(first, second); }
Function operator>(ExpressionView first, const Tensor& second) { return expression::gt(first, second); }
Function operator>(ExpressionView first, Param second) { return expression::gt(first, second); }
Function operator>(const Tensor& first, ExpressionView second) { return expression::gt(first, second); }
Function operator>(Param first, ExpressionView second) { return expression::gt(first, second); }

Function operator==(ExpressionView first, ExpressionView second) { return expression::eq(first, second); }
Function operator==(ExpressionView first, const Tensor& second) { return expression::eq(first, second); }
Function operator==(ExpressionView first, Param second) { return expression::eq(first, second); }
Function operator==(const Tensor& first, ExpressionView second) { return expression::eq(first, second); }
Function operator==(Param first, ExpressionView second) { return expression::eq(first, second); }

Function operator!=(ExpressionView first, ExpressionView second) { return expression::ne(first, second); }
Function operator!=(ExpressionView first, const Tensor& second) { return expression::ne(first, second); }
Function operator!=(ExpressionView first, Param second) { return expression::ne(first, second); }
Function operator!=(const Tensor& first, ExpressionView second) { return expression::ne(first, second); }
Function operator!=(Param first, ExpressionView second) { return expression::ne(first, second); }

Function operator<=(ExpressionView first, ExpressionView second) { return expression::le(first, second); }
Function operator<=(ExpressionView first, const Tensor& second) { return expression::le(first, second); }
Function operator<=(ExpressionView first, Param second) { return expression::le(first, second); }
Function operator<=(const Tensor& first, ExpressionView second) { return expression::le(first, second); }
Function operator<=(Param first, ExpressionView second) { return expression::le(first, second); }

Function operator>=(ExpressionView first, ExpressionView second) { return expression::ge(first, second); }
Function operator>=(ExpressionView first, const Tensor& second) { return expression::ge(first, second); }
Function operator>=(ExpressionView first, Param second) { return expression::ge(first, second); }
Function operator>=(const Tensor& first, ExpressionView second) { return expression::ge(first, second); }
Function operator>=(Param first, ExpressionView second) { return expression::ge(first, second); }

Function operator and(ExpressionView first, ExpressionView second) { return expression::logical_and(first, second); }
Function operator and(ExpressionView first, const Tensor& second) { return expression::logical_and(first, second); }
Function operator and(ExpressionView first, Param second) { return expression::logical_and(first, second); }
Function operator and(const Tensor& first, ExpressionView second) { return expression::logical_and(first, second); }
Function operator and(Param first, ExpressionView second) { return expression::logical_and(first, second); }

Function operator or(ExpressionView first, ExpressionView second) { return expression::logical_or(first, second); }
Function operator or(ExpressionView first, const Tensor& second) { return expression::logical_or(first, second); }
Function operator or(ExpressionView first, Param second) { return expression::logical_or(first, second); }
Function operator or(const Tensor& first, ExpressionView second) { return expression::logical_or(first, second); }
Function operator or(Param first, ExpressionView second) { return expression::logical_or(first, second); }

Function UnaryFn::operator()(ExpressionView first) const { return expression::defer(operation, first).value(); }

Function UnaryWithParamFn::operator()(ExpressionView first, Param second) const {
    return expression::defer(operation, first, second).value();
}

Function RUnaryWithParamFn::operator()(Param first, ExpressionView second) const {
    return expression::defer(operation, second, first).value();
}

Function BinaryFn::operator()(ExpressionView first, ExpressionView second) const {
    return expression::defer(operation, first, second).value();
}

Function RBinaryFn::operator()(ExpressionView first, ExpressionView second) const {
    return expression::defer(operation, second, first).value();
}

Function TernaryFn::operator()(ExpressionView first, ExpressionView second, ExpressionView third) const {
    return expression::defer(operation, first, second, third).value();
}

Function DivFn::operator()(Param first, ExpressionView second) const { return first * expression::reciprocal(second); }

Function RDivFn::operator()(ExpressionView first, Param second) const { return second / first; }

Function CompareFn::operator()(ExpressionView first, ExpressionView second) const { return operation(first - second); }

Function CompareFn::operator()(ExpressionView first, Param second) const { return operation(first - second); }

Function CompareFn::operator()(Param first, ExpressionView second) const { return operation(first - second); }

inline auto _nez(ExpressionView first) { return expression::nez(first); }

constexpr auto _nez(Param first) {
    return std::visit([](auto value) { return value != 0; }, first);
}

template <typename First, typename Second>
auto _logical_binary(OverloadedBinaryFn operation, First first, Second second) {
    return expression::nez(operation(expression::_nez(first), expression::_nez(second)));
}

Function LogicalBinaryFn::operator()(ExpressionView first, ExpressionView second) const {
    return expression::_logical_binary(operation, first, second);
}

Function LogicalBinaryFn::operator()(ExpressionView first, Param second) const {
    return expression::_logical_binary(operation, first, second);
}

Function LogicalBinaryFn::operator()(Param first, ExpressionView second) const {
    return expression::_logical_binary(operation, first, second);
}

template <typename Input, typename Other>
    requires std::constructible_from<Param, Input> or std::constructible_from<Param, Other>
auto _where(ExpressionView condition, const Input& input, const Other& other) {
    return (expression::nez(condition) * input) + (expression::eqz(condition) * other);
}

Function WhereFn::operator()(ExpressionView condition, ExpressionView input, Param other) const {
    return expression::_where(condition, input, other);
}

Function WhereFn::operator()(ExpressionView condition, Param input, ExpressionView other) const {
    return expression::_where(condition, input, other);
}

Function WhereFn::operator()(ExpressionView condition, Param input, Param other) const {
    return expression::_where(condition, input, other);
}

inline auto _ltz(ExpressionView first) { return expression::ltz(first); }

constexpr auto _ltz(Param first) {
    return std::visit([](auto value) { return value < 0; }, first);
}

inline auto _gtz(ExpressionView first) { return expression::gtz(first); }

constexpr auto _gtz(Param first) {
    return std::visit([](auto value) { return value > 0; }, first);
}

inline auto _gez(ExpressionView first) { return expression::gez(first); }

constexpr auto _gez(Param first) {
    return std::visit([](auto value) { return value >= 0; }, first);
}

inline auto _where(ExpressionView condition, ExpressionView input, ExpressionView other) {
    return expression::where(condition, input, other);
}

template <typename Input, typename Other>
constexpr auto _where(bool condition, const Input& input, const Other& other) {
    return condition ? input : other;
}

template <typename Condition, typename Input, typename... Other>
    requires(sizeof...(Other) > 1)
constexpr auto _where(const Condition& condition, const Input& input, const Other&... other) {
    return expression::_where(condition, input, expression::_where(other...));
}

template <typename First, typename Second>
auto _atan2(First y, Second x) {
    constexpr auto pi = std::numbers::pi_v<float>;
    const auto atan = expression::atan(y / x);
    return expression::_where(
        expression::_gtz(x),
        atan,
        expression::_ltz(x) and expression::_gez(y),
        atan + pi,
        expression::_ltz(x) and expression::_ltz(y),
        atan - pi,
        expression::_gtz(y),
        pi / 2,
        expression::_ltz(y),
        -pi / 2,
        0);
}

Function Atan2Fn::operator()(ExpressionView first, Param second) const { return expression::_atan2(first, second); }

Function Atan2Fn::operator()(Param first, ExpressionView second) const { return expression::_atan2(first, second); }

Function Atan2Fn::operator()(ExpressionView first, ExpressionView second) const {
    return expression::_atan2(first, second);
}

}  // namespace ttnn::operations::expression
