// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/lazy/node.hpp"
#include "ttnn/operations/eltwise/lazy/operation.hpp"
#include "ttnn/operations/eltwise/lazy/param.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include <span>
#include <variant>
#include <vector>

namespace ttnn::operations::lazy {

template <typename>
class BasicExpressionView;

using ExpressionView = BasicExpressionView<Node>;
using FunctionView = BasicExpressionView<FunctionNode>;

using Value = std::variant<Tensor, FunctionView>;

template <typename>
class BasicExpression;

using Expression = BasicExpression<Node>;
using Function = BasicExpression<FunctionNode>;

template <typename>
class BasicExpressionView {
    friend ExpressionView;
    friend FunctionView;
    friend Expression;
    friend Function;

    std::span<const Node> nodes;
    bool is_root;

    BasicExpressionView(std::span<const Node> nodes, bool is_root) noexcept;

    const Node& root() const noexcept;

    const FunctionNode& fnode() const noexcept
        requires std::same_as<BasicExpressionView, FunctionView>;

public:
    // do not allow Expression to convert to FunctionView
    BasicExpressionView(const Expression& expression) noexcept
        requires std::same_as<BasicExpressionView, ExpressionView>;

    BasicExpressionView(const Function& function) noexcept;

    BasicExpressionView(const BasicExpressionView&) noexcept = default;

    // FunctionView can implicitly widen to ExpressionView
    BasicExpressionView(const FunctionView& function) noexcept
        requires std::same_as<BasicExpressionView, ExpressionView>;

    // getters for ExpressionView

    [[nodiscard]] std::optional<Tensor> tensor() const
        requires std::same_as<BasicExpressionView, ExpressionView>;
    [[nodiscard]] std::optional<FunctionView> function() const noexcept
        requires std::same_as<BasicExpressionView, ExpressionView>;
    [[nodiscard]] Value value() const
        requires std::same_as<BasicExpressionView, ExpressionView>;

    // getters for FunctionView

    [[nodiscard]] Operation operation() const noexcept
        requires std::same_as<BasicExpressionView, FunctionView>;
    [[nodiscard]] Arguments<ExpressionView> arguments() const noexcept
        requires std::same_as<BasicExpressionView, FunctionView>;
    [[nodiscard]] ParamsView params() const noexcept
        requires std::same_as<BasicExpressionView, FunctionView>;

    // getters for both

    [[nodiscard]] DataType dtype() const noexcept;
    [[nodiscard]] const Shape& logical_shape() const noexcept;
    [[nodiscard]] tt::CBIndex cb_index() const noexcept;
    [[nodiscard]] std::size_t rt_offset() const noexcept;
    [[nodiscard]] std::size_t inputs() const noexcept;
    [[nodiscard]] std::size_t circular_buffers() const noexcept;
};

// Unary
[[nodiscard]] Function operator-(ExpressionView first);

// Binary
[[nodiscard]] Function operator+(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator+(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator+(ExpressionView first, Param second);
[[nodiscard]] Function operator+(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator+(Param first, ExpressionView second);

[[nodiscard]] Function operator-(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator-(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator-(ExpressionView first, Param second);
[[nodiscard]] Function operator-(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator-(Param first, ExpressionView second);

[[nodiscard]] Function operator*(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator*(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator*(ExpressionView first, Param second);
[[nodiscard]] Function operator*(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator*(Param first, ExpressionView second);

[[nodiscard]] Function operator/(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator/(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator/(ExpressionView first, Param second);
[[nodiscard]] Function operator/(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator/(Param first, ExpressionView second);

[[nodiscard]] Function operator<(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator<(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator<(ExpressionView first, Param second);
[[nodiscard]] Function operator<(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator<(Param first, ExpressionView second);

[[nodiscard]] Function operator>(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator>(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator>(ExpressionView first, Param second);
[[nodiscard]] Function operator>(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator>(Param first, ExpressionView second);

[[nodiscard]] Function operator==(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator==(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator==(ExpressionView first, Param second);
[[nodiscard]] Function operator==(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator==(Param first, ExpressionView second);

[[nodiscard]] Function operator!=(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator!=(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator!=(ExpressionView first, Param second);
[[nodiscard]] Function operator!=(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator!=(Param first, ExpressionView second);

[[nodiscard]] Function operator<=(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator<=(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator<=(ExpressionView first, Param second);
[[nodiscard]] Function operator<=(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator<=(Param first, ExpressionView second);

[[nodiscard]] Function operator>=(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator>=(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator>=(ExpressionView first, Param second);
[[nodiscard]] Function operator>=(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator>=(Param first, ExpressionView second);

[[nodiscard]] Function operator and(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator and(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator and(ExpressionView first, Param second);
[[nodiscard]] Function operator and(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator and(Param first, ExpressionView second);

[[nodiscard]] Function operator or(ExpressionView first, ExpressionView second);
[[nodiscard]] Function operator or(ExpressionView first, const Tensor& second);
[[nodiscard]] Function operator or(ExpressionView first, Param second);
[[nodiscard]] Function operator or(const Tensor& first, ExpressionView second);
[[nodiscard]] Function operator or(Param first, ExpressionView second);

// traverses in post-order
template <typename Visitor>
void traverse(Visitor visitor, ExpressionView expression) {
    std::visit(
        ttsl::overloaded{
            [&](const Tensor& tensor) { visitor(tensor); },
            [&](FunctionView function) {
                for (auto argument : function.arguments()) {
                    traverse<Visitor&>(visitor, argument);
                }

                visitor(function);
            }},
        expression.value());
}

template <typename>
class BasicExpression {
    friend ExpressionView;
    friend FunctionView;
    friend Expression;

    std::vector<Node> nodes;

    // Function must not be constructible from Tensor
    explicit BasicExpression(const Tensor& tensor)
        requires std::same_as<BasicExpression, Expression>;

    explicit BasicExpression(Unary operation, ExpressionView first)
        requires std::same_as<BasicExpression, Function>;

    explicit BasicExpression(UnaryWithParam operation, ExpressionView first, Param second)
        requires std::same_as<BasicExpression, Function>;

    explicit BasicExpression(Binary operation, ExpressionView first, ExpressionView second)
        requires std::same_as<BasicExpression, Function>;

    explicit BasicExpression(Ternary operation, ExpressionView first, ExpressionView second, ExpressionView third)
        requires std::same_as<BasicExpression, Function>;

public:
    // Function must not be constructible from Tensor
    [[nodiscard]] static std::optional<BasicExpression> from(const Tensor& tensor)
        requires std::same_as<BasicExpression, Expression>;

    [[nodiscard]] static std::optional<BasicExpression> from(Unary operation, ExpressionView first)
        requires std::same_as<BasicExpression, Function>;

    [[nodiscard]] static std::optional<BasicExpression> from(
        UnaryWithParam operation, ExpressionView first, Param second)
        requires std::same_as<BasicExpression, Function>;

    [[nodiscard]] static std::optional<BasicExpression> from(
        Binary operation, ExpressionView first, ExpressionView second)
        requires std::same_as<BasicExpression, Function>;

    [[nodiscard]] static std::optional<BasicExpression> from(
        Ternary operation, ExpressionView first, ExpressionView second, ExpressionView third)
        requires std::same_as<BasicExpression, Function>;

    BasicExpression(const BasicExpression&) = default;
    BasicExpression(BasicExpression&&) noexcept = default;

    // Function can implicitly widen to Expression
    BasicExpression(const Function& function)
        requires std::same_as<BasicExpression, Expression>;
    BasicExpression(Function&& function) noexcept
        requires std::same_as<BasicExpression, Expression>;

    // getters for Expression

    [[nodiscard]] std::optional<Tensor> tensor() const
        requires std::same_as<BasicExpression, Expression>;
    [[nodiscard]] std::optional<FunctionView> function() const noexcept
        requires std::same_as<BasicExpression, Expression>;
    [[nodiscard]] Value value() const
        requires std::same_as<BasicExpression, Expression>;

    // getters for Function

    [[nodiscard]] Operation operation() const noexcept
        requires std::same_as<BasicExpression, Function>;
    [[nodiscard]] Arguments<ExpressionView> arguments() const noexcept
        requires std::same_as<BasicExpression, Function>;
    [[nodiscard]] ParamsView params() const noexcept
        requires std::same_as<BasicExpression, Function>;

    // getters for both

    [[nodiscard]] DataType dtype() const noexcept;
    [[nodiscard]] const Shape& logical_shape() const noexcept;
    [[nodiscard]] tt::CBIndex cb_index() const noexcept;
    [[nodiscard]] std::size_t rt_offset() const noexcept;
    [[nodiscard]] std::size_t inputs() const noexcept;
    [[nodiscard]] std::size_t circular_buffers() const noexcept;
};

std::optional<Expression> defer(const Tensor& tensor);

std::optional<Function> defer(Unary unary, ExpressionView first);

std::optional<Function> defer(UnaryWithParam unary, ExpressionView first, Param second);

std::optional<Function> defer(Binary binary, ExpressionView first, ExpressionView second);

std::optional<Function> defer(Ternary ternary, ExpressionView first, ExpressionView second, ExpressionView third);

std::string to_compute_kernel_string(FunctionView expression);

std::string to_debug_string(FunctionView expression);

}  // namespace ttnn::operations::lazy
