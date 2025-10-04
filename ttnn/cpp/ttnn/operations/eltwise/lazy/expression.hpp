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

    BasicExpressionView(std::span<const Node> nodes) noexcept;

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

    std::optional<Tensor> tensor() const
        requires std::same_as<BasicExpressionView, ExpressionView>;
    std::optional<FunctionView> function() const noexcept
        requires std::same_as<BasicExpressionView, ExpressionView>;
    Value value() const
        requires std::same_as<BasicExpressionView, ExpressionView>;

    // getters for FunctionView

    Operation operation() const noexcept
        requires std::same_as<BasicExpressionView, FunctionView>;
    Arguments<ExpressionView> arguments() const noexcept
        requires std::same_as<BasicExpressionView, FunctionView>;
    ParamsView params() const noexcept
        requires std::same_as<BasicExpressionView, FunctionView>;

    // getters for both

    DataType dtype() const noexcept;
    const Shape& logical_shape() const noexcept;
    tt::CBIndex index() const noexcept;
};

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
        expression);
}

template <typename>
class BasicExpression {
    friend ExpressionView;
    friend FunctionView;

    std::vector<Node> nodes;

    // Function must not be constructible from Tensor
    explicit BasicExpression(const Tensor& tensor)
        requires std::same_as<BasicExpression, Expression>;

    explicit BasicExpression(Unary operation, ExpressionView first, std::initializer_list<Param> params);

    explicit BasicExpression(
        Binary operation, ExpressionView first, ExpressionView second, std::initializer_list<Param> params);

    explicit BasicExpression(
        Ternary operation,
        ExpressionView first,
        ExpressionView second,
        ExpressionView third,
        std::initializer_list<Param> params);

public:
    // Function must not be constructible from Tensor
    static std::optional<BasicExpression> from(const Tensor& tensor)
        requires std::same_as<BasicExpression, Expression>;

    static std::optional<BasicExpression> from(
        Unary operation, ExpressionView first, std::initializer_list<Param> params);

    static std::optional<BasicExpression> from(
        Binary operation, ExpressionView first, ExpressionView second, std::initializer_list<Param> params);

    static std::optional<BasicExpression> from(
        Ternary operation,
        ExpressionView first,
        ExpressionView second,
        ExpressionView third,
        std::initializer_list<Param> params);

    // getters for Expression

    std::optional<Tensor> tensor() const
        requires std::same_as<BasicExpression, Expression>;
    std::optional<FunctionView> function() const noexcept
        requires std::same_as<BasicExpression, Expression>;
    Value value() const
        requires std::same_as<BasicExpression, Expression>;

    // getters for Function

    Operation operation() const noexcept
        requires std::same_as<BasicExpression, Function>;
    Arguments<ExpressionView> arguments() const noexcept
        requires std::same_as<BasicExpression, Function>;
    ParamsView params() const noexcept
        requires std::same_as<BasicExpression, Function>;

    // getters for both

    DataType dtype() const noexcept;
    const Shape& logical_shape() const noexcept;
    tt::CBIndex index() const noexcept;
};

std::string to_compute_kernel_string(ExpressionView expression);

}  // namespace ttnn::operations::lazy
