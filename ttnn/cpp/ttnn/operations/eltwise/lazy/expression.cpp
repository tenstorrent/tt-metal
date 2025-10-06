// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_stl/assert.hpp"
#include "tt_stl/overloaded.hpp"
#include "ttnn/operations/eltwise/lazy/expression.hpp"

#include <enchantum/enchantum.hpp>
#include <fmt/ranges.h>

namespace ttnn::operations::lazy {

template <typename T>
BasicExpressionView<T>::BasicExpressionView(std::span<const Node> nodes) noexcept : nodes(nodes) {}

template <typename T>
BasicExpressionView<T>::BasicExpressionView(const Expression& expression) noexcept
    requires std::same_as<BasicExpressionView, ExpressionView>
    : nodes(expression.nodes) {}

template <typename T>
BasicExpressionView<T>::BasicExpressionView(const Function& function) noexcept : nodes(function.nodes) {}

template <typename T>
BasicExpressionView<T>::BasicExpressionView(const FunctionView& function) noexcept
    requires std::same_as<BasicExpressionView, ExpressionView>
    : nodes(function.nodes) {}

template <typename T>
const Node& BasicExpressionView<T>::root() const noexcept {
    // Expressions are correct by construction
    // assume nodes is non-empty
    return nodes.back();
}

template <typename T>
const FunctionNode& BasicExpressionView<T>::fnode() const noexcept
    requires std::same_as<BasicExpressionView, FunctionView>
{
    // FunctionView is correct by construction
    // assume root is FunctionNode
    return *std::get_if<FunctionNode>(&root());
}

template <typename T>
BasicExpression<T>::BasicExpression(const Tensor& tensor)
    requires std::same_as<BasicExpression, Expression>
    : nodes{tensor} {}

template <typename T>
BasicExpression<T>::BasicExpression(Unary operation, ExpressionView first, Params&& params) {
    nodes.reserve(first.nodes.size() + 1);
    nodes.insert(nodes.end(), first.nodes.begin(), first.nodes.end());
    nodes.emplace_back(FunctionNode{.operation = operation, .offsets = {1}, .params = std::move(params)});
}

template <typename T>
BasicExpression<T>::BasicExpression(Binary operation, ExpressionView first, ExpressionView second, Params&& params) {
    nodes.reserve(first.nodes.size() + second.nodes.size() + 1);
    nodes.insert(nodes.end(), first.nodes.begin(), first.nodes.end());
    nodes.insert(nodes.end(), second.nodes.begin(), second.nodes.end());
    nodes.emplace_back(
        FunctionNode{.operation = operation, .offsets = {second.nodes.size() + 1, 1}, .params = std::move(params)});
}

template <typename T>
BasicExpression<T>::BasicExpression(
    Ternary operation, ExpressionView first, ExpressionView second, ExpressionView third, Params&& params) {
    nodes.reserve(first.nodes.size() + second.nodes.size() + third.nodes.size() + 1);
    nodes.insert(nodes.end(), first.nodes.begin(), first.nodes.end());
    nodes.insert(nodes.end(), second.nodes.begin(), second.nodes.end());
    nodes.insert(nodes.end(), third.nodes.begin(), third.nodes.end());
    nodes.emplace_back(FunctionNode{
        .operation = operation,
        .offsets = {second.nodes.size() + third.nodes.size() + 1, third.nodes.size() + 1, 1},
        .params = std::move(params)});
}

template <typename T>
std::optional<BasicExpression<T>> BasicExpression<T>::from(const Tensor& tensor)
    requires std::same_as<BasicExpression, Expression>
{
    if (tensor.storage_type() != StorageType::DEVICE) {
        return std::nullopt;
    }

    if (not tensor.is_allocated()) {
        return std::nullopt;
    }

    // add tensor validation here

    return BasicExpression<T>(tensor);
}

template <typename T>
std::optional<BasicExpression<T>> BasicExpression<T>::from(Unary operation, ExpressionView first, Params params) {
    // add unary operation validation here

    return BasicExpression<T>(operation, first, std::move(params));
}

template <typename T>
std::optional<BasicExpression<T>> BasicExpression<T>::from(
    Binary operation, ExpressionView first, ExpressionView second, Params params) {
    if (first.dtype() != second.dtype()) {
        return std::nullopt;
    }

    if (first.logical_shape() != second.logical_shape()) {
        return std::nullopt;
    }

    if (first.index() + second.index() + 2 >= tt::CBIndex::SIZE) {
        return std::nullopt;
    }

    // add binary operation validation here

    return BasicExpression<T>(operation, first, second, std::move(params));
}

template <typename T>
std::optional<BasicExpression<T>> BasicExpression<T>::from(
    Ternary operation, ExpressionView first, ExpressionView second, ExpressionView third, Params params) {
    if (first.dtype() != second.dtype() or first.dtype() != third.dtype()) {
        return std::nullopt;
    }

    if (first.logical_shape() != second.logical_shape() or first.logical_shape() != third.logical_shape()) {
        return std::nullopt;
    }

    if (first.index() + second.index() + third.index() + 3 >= tt::CBIndex::SIZE) {
        return std::nullopt;
    }

    // add ternary operation validation here

    return BasicExpression<T>(operation, first, second, third, std::move(params));
}

// obtains left-most tensor in expression tree
const Tensor& get_tensor(std::span<const Node> nodes) noexcept {
    auto index = nodes.size() - 1;

    while (auto fnode_ptr = std::get_if<FunctionNode>(&nodes[index])) {
        index -= fnode_ptr->offsets.front();
    }

    static_assert(std::variant_size_v<Node> == 2, "assumption below relies on process of elimination");
    // must be Tensor if not FunctionNode
    return *std::get_if<Tensor>(&nodes[index]);
}

template <typename T>
DataType BasicExpressionView<T>::dtype() const noexcept {
    return get_tensor(nodes).dtype();
}

template <typename T>
DataType BasicExpression<T>::dtype() const noexcept {
    return get_tensor(nodes).dtype();
}

template <typename T>
const Shape& BasicExpressionView<T>::logical_shape() const noexcept {
    return get_tensor(nodes).logical_shape();
}

template <typename T>
const Shape& BasicExpression<T>::logical_shape() const noexcept {
    return get_tensor(nodes).logical_shape();
}

// computes the circular buffer index accounting for re-use between internal nodes
template <typename T>
tt::CBIndex BasicExpressionView<T>::index() const noexcept {
    std::uint8_t total_indices = 0;
    std::uint8_t reusable_indices = 0;

    for (std::size_t index = 0; const auto& node : nodes) {
        std::visit(
            ttsl::overloaded{
                [&](const Tensor&) { ++total_indices; },
                [&](const FunctionNode& function) {
                    for (const auto offset : function.offsets) {
                        if (std::holds_alternative<FunctionNode>(nodes[index - offset])) {
                            // this assumes no common sub-expressions
                            // when CSE is introduced, this will need revised
                            ++reusable_indices;
                        }
                    }

                    if (reusable_indices > 0) {
                        --reusable_indices;
                    } else {
                        ++total_indices;
                    }
                }},
            node);

        ++index;
    }

    return tt::CBIndex(total_indices - reusable_indices - 1);
}

template <typename T>
tt::CBIndex BasicExpression<T>::index() const noexcept {
    return ExpressionView(*this).index();
}

template <typename T>
std::optional<Tensor> BasicExpressionView<T>::tensor() const
    requires std::same_as<BasicExpressionView, ExpressionView>
{
    if (const auto tensor_ptr = std::get_if<Tensor>(&root())) {
        return *tensor_ptr;
    }

    return std::nullopt;
}

template <typename T>
std::optional<FunctionView> BasicExpressionView<T>::function() const noexcept
    requires std::same_as<BasicExpressionView, ExpressionView>
{
    if (std::holds_alternative<FunctionNode>(root())) {
        return FunctionView(nodes);
    }

    return std::nullopt;
}

template <typename T>
Value BasicExpressionView<T>::value() const
    requires std::same_as<BasicExpressionView, ExpressionView>
{
    if (const auto tensor_ptr = std::get_if<Tensor>(&root())) {
        return Value(*tensor_ptr);
    }

    static_assert(std::variant_size_v<Value> == 2, "assumption below relies on process of elimination");
    // must be FunctionView if not Tensor
    return Value(FunctionView(nodes));
}

template <typename T>
std::optional<Tensor> BasicExpression<T>::tensor() const
    requires std::same_as<BasicExpression, Expression>
{
    return ExpressionView(*this).tensor();
}

template <typename T>
std::optional<FunctionView> BasicExpression<T>::function() const noexcept
    requires std::same_as<BasicExpression, Expression>
{
    return ExpressionView(*this).function();
}

template <typename T>
Value BasicExpression<T>::value() const
    requires std::same_as<BasicExpression, Expression>
{
    return ExpressionView(*this).value();
}

template <typename T>
Operation BasicExpressionView<T>::operation() const noexcept
    requires std::same_as<BasicExpressionView, FunctionView>
{
    return fnode().operation;
}

template <typename T>
Arguments<ExpressionView> BasicExpressionView<T>::arguments() const noexcept
    requires std::same_as<BasicExpressionView, FunctionView>
{
    Arguments<ExpressionView> result;

    for (const auto offset : fnode().offsets) {
        result.push_back(ExpressionView({nodes.begin(), nodes.size() - offset}));
    }

    return result;
}

template <typename T>
ParamsView BasicExpressionView<T>::params() const noexcept
    requires std::same_as<BasicExpressionView, FunctionView>
{
    return fnode().params;
}

template <typename T>
Operation BasicExpression<T>::operation() const noexcept
    requires std::same_as<BasicExpression, Function>
{
    return FunctionView(*this).operation();
}

template <typename T>
Arguments<ExpressionView> BasicExpression<T>::arguments() const noexcept
    requires std::same_as<BasicExpression, Function>
{
    return FunctionView(*this).arguments();
}

template <typename T>
ParamsView BasicExpression<T>::params() const noexcept
    requires std::same_as<BasicExpression, Function>
{
    return FunctionView(*this).params();
}

template class BasicExpressionView<Node>;
template class BasicExpressionView<FunctionNode>;
template class BasicExpression<Node>;
template class BasicExpression<FunctionNode>;

static_assert(std::is_trivially_copyable_v<ExpressionView>);
static_assert(std::is_trivially_copyable_v<FunctionView>);

std::string i32_u16_u32_function_name(DataType dtype, std::string operation) {
    using enum DataType;
    switch (dtype) {
        case INT32: return fmt::format("{}_int32", operation);
        case UINT16: return fmt::format("{}_uint16", operation);
        case UINT32: return fmt::format("{}_uint32", operation);
        default: return operation;
    }
}

std::string i32_u16_function_name(DataType dtype, std::string operation) {
    using enum DataType;
    switch (dtype) {
        case INT32: return fmt::format("{}_int32", operation);
        case UINT16: return fmt::format("{}_uint16", operation);
        default: return operation;
    }
}

std::string i32_function_name(DataType dtype, std::string operation) {
    using enum DataType;
    switch (dtype) {
        case INT32: return fmt::format("{}_int32", operation);
        default: return operation;
    }
}

std::string f32_i32_function_name(DataType dtype, std::string operation) {
    using enum DataType;
    switch (dtype) {
        case FLOAT32: return fmt::format("{}_fp32", operation);
        case INT32: return fmt::format("{}_int32", operation);
        default: return operation;
    }
}

std::string function_name(DataType dtype, Unary operation) {
    using enum Unary;
    switch (operation) {
        case ADD: return i32_function_name(dtype, "add_unary");
        case SUB: return "sub_unary";
        case RSUB: return "rsub_unary";
        case MUL: return "mul_unary";
        case DIV: return "div_unary";
        case NEGATIVE: return i32_function_name(dtype, "negative");
        case EXP: return "exp";
        case POWER: return "power";
        case EQZ: return i32_u16_u32_function_name(dtype, "eqz");
        case GEZ: return i32_function_name(dtype, "gez");
        case GTZ: return i32_function_name(dtype, "gtz");
        case LEZ: return i32_function_name(dtype, "lez");
        case LTZ: return i32_function_name(dtype, "ltz");
        case NEZ: return i32_u16_u32_function_name(dtype, "nez");
        case LOGICAL_NOT: return i32_u16_u32_function_name(dtype, "logical_not_unary");
    }
}

std::string function_name(DataType dtype, Binary operation) {
    using enum Binary;
    switch (operation) {
        case ADD: return "add";
        case SUB: return "sub";
        case MUL: return "mul";
    }
}

std::string function_name(DataType dtype, Ternary operation) {
    using enum Ternary;
    switch (operation) {
        case WHERE: return f32_i32_function_name(dtype, "where");
    }
}

// TODO consider handling copy on kernel-side to simplify host-side logic
void format_to(
    std::back_insert_iterator<std::string> out, DataType dtype, Unary operation, ParamsView params, int& rt_arg_index) {
    switch (params.size()) {
        case 0:
            fmt::format_to(out, "views::copy<0>(0,0)|views::{}(0))|views::pack<1>(0)", function_name(dtype, operation));
            return;
        case 1:
            fmt::format_to(
                out,
                "views::copy<0>(0,0)|views::{}(0,get_arg_val<uint32_t>({}))|views::pack<1>(0)",
                function_name(dtype, operation),
                rt_arg_index++);
            return;
        default:
            TT_THROW(
                "unexpected params size {} for unary {} {}",
                params.size(),
                enchantum::to_string(dtype),
                enchantum::to_string(operation));
    }
}

void format_to(
    std::back_insert_iterator<std::string> out,
    DataType dtype,
    Binary operation,
    ParamsView params,
    int& rt_arg_index) {
    switch (params.size()) {
        case 0:
            fmt::format_to(
                out, "views::{}<0,1>(0,0,0)|views::pack<2>(0)", function_name(dtype, operation), rt_arg_index++);
            return;
        default:
            TT_THROW(
                "unexpected params size {} for binary {} {}",
                params.size(),
                enchantum::to_string(dtype),
                enchantum::to_string(operation));
    }
}

void format_to(
    std::back_insert_iterator<std::string> out,
    DataType dtype,
    Ternary operation,
    ParamsView params,
    int& rt_arg_index) {
    switch (params.size()) {
        case 0:
            fmt::format_to(
                out,
                "views::copy<0>(0,0)|views::copy<1>(0,n_tiles)|views::copy<2>(0,n_tiles*2)|views::{}(0,n_tiles,n_tiles*"
                "2,0)|views::pack<3>(0)",
                function_name(dtype, operation),
                rt_arg_index++);
            return;
        default:
            TT_THROW(
                "unexpected params size {} for ternary {} {}",
                params.size(),
                enchantum::to_string(dtype),
                enchantum::to_string(operation));
    }
}

void format_to(
    std::back_insert_iterator<std::string> out,
    DataType dtype,
    Operation operation,
    ParamsView params,
    int& rt_arg_index) {
    return std::visit(
        [&](auto alternative) { return lazy::format_to(out, dtype, alternative, params, rt_arg_index); }, operation);
}

void format_to(std::back_insert_iterator<std::string> out, FunctionView function, int& rt_arg_index) {
    auto cb_ids = fmt::join(
        function.arguments() | std::views::transform([](ExpressionView expression) {
            return fmt::format("tt::{}", enchantum::to_string(expression.index()));
        }),
        ",");

    // passing views as lvalues is disallowed
    // NOLINTNEXTLINE(performance-move-const-arg)
    fmt::format_to(out, "views::with_cb_ids<{},tt::{}>(", std::move(cb_ids), enchantum::to_string(function.index()));
    lazy::format_to(out, function.dtype(), function.operation(), function.params(), rt_arg_index);
    *out++ = ')';
}

std::string to_compute_kernel_string(ExpressionView expression) {
    std::string result;
    // n_tiles is index 0
    int rt_arg_index = 1;
    int tensor_count = 1;  // always output tensor
    std::uint8_t total_indices = 0;
    std::uint8_t reusable_indices = 0;

    ttnn::operations::lazy::traverse(
        ttsl::overloaded{
            [&](const Tensor&) { ++tensor_count; },
            [&](FunctionView function) {
                for (const auto expression : function.arguments()) {
                    if (expression.function()) {
                        // this assumes no common sub-expressions
                        // when CSE is introduced, this will need revised
                        ++reusable_indices;
                    }
                }

                if (reusable_indices > 0) {
                    --reusable_indices;
                } else {
                    ++total_indices;
                }
            }},
        expression);

    const auto append_prefix = [&] {
        std::format_to(
            std::back_inserter(result),
            "using "
            "View=views::MakeComputeView<{},{}>;View::init_tiles(views::init_sfpu<tt::c_0,tt::{}>());View::compute_"
            "tiles(n_tiles,",
            tensor_count,
            total_indices,
            enchantum::to_string(tt::CBIndex(tensor_count - 1)));
    };

    ttnn::operations::lazy::traverse(
        ttsl::overloaded{
            [](const ttnn::Tensor&) {},
            [&](FunctionView function) {
                // prefix or separator
                if (result.empty()) {
                    append_prefix();
                } else {
                    result.push_back('|');
                }

                format_to(std::back_inserter(result), function, rt_arg_index);
            }},
        expression);

    format_to(std::back_inserter(result), ");");

    return result;
}

}  // namespace ttnn::operations::lazy
