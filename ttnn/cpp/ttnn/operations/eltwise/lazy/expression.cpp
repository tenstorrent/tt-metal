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
BasicExpressionView<T>::BasicExpressionView(std::span<const Node> nodes, bool is_root) noexcept :
    nodes(nodes), is_root(is_root) {}

template <typename T>
BasicExpressionView<T>::BasicExpressionView(const Expression& expression) noexcept
    requires std::same_as<BasicExpressionView, ExpressionView>
    : nodes(expression.nodes), is_root(true) {}

template <typename T>
BasicExpressionView<T>::BasicExpressionView(const Function& function) noexcept : nodes(function.nodes), is_root(true) {}

template <typename T>
BasicExpressionView<T>::BasicExpressionView(const FunctionView& function) noexcept
    requires std::same_as<BasicExpressionView, ExpressionView>
    : nodes(function.nodes), is_root(function.is_root) {}

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

// obtains postfix subnodes starting from left-most leaf node of subexpression
std::span<const Node> get_subnodes(std::span<const Node> nodes) noexcept {
    auto node = &nodes.back();

    static_assert(std::variant_size_v<Node> == 2, "assumption below relies on process of elimination");
    // FunctionNode is only possible internal node
    while (auto fnode_ptr = std::get_if<FunctionNode>(node)) {
        node -= fnode_ptr->offsets.front();
    }

    return {node, std::to_address(nodes.end())};
}

template <typename T>
BasicExpression<T>::BasicExpression(Unary operation, ExpressionView first, Params&& params)
    requires std::same_as<BasicExpression, Function>
{
    const auto first_nodes = get_subnodes(first.nodes);
    nodes.reserve(first_nodes.size() + 1);
    nodes.insert(nodes.end(), first_nodes.begin(), first_nodes.end());
    nodes.emplace_back(FunctionNode{.operation = operation, .offsets = {1}, .params = std::move(params)});
}

template <typename T>
BasicExpression<T>::BasicExpression(Binary operation, ExpressionView first, ExpressionView second, Params&& params)
    requires std::same_as<BasicExpression, Function>
{
    const auto first_nodes = get_subnodes(first.nodes);
    const auto second_nodes = get_subnodes(second.nodes);
    nodes.reserve(first_nodes.size() + second_nodes.size() + 1);
    nodes.insert(nodes.end(), first_nodes.begin(), first_nodes.end());
    nodes.insert(nodes.end(), second_nodes.begin(), second_nodes.end());
    nodes.emplace_back(
        FunctionNode{.operation = operation, .offsets = {second_nodes.size() + 1, 1}, .params = std::move(params)});
}

template <typename T>
BasicExpression<T>::BasicExpression(
    Ternary operation, ExpressionView first, ExpressionView second, ExpressionView third, Params&& params)
    requires std::same_as<BasicExpression, Function>
{
    const auto first_nodes = get_subnodes(first.nodes);
    const auto second_nodes = get_subnodes(second.nodes);
    const auto third_nodes = get_subnodes(third.nodes);
    nodes.reserve(first_nodes.size() + second_nodes.size() + third_nodes.size() + 1);
    nodes.insert(nodes.end(), first_nodes.begin(), first_nodes.end());
    nodes.insert(nodes.end(), second_nodes.begin(), second_nodes.end());
    nodes.insert(nodes.end(), third_nodes.begin(), third_nodes.end());
    nodes.emplace_back(FunctionNode{
        .operation = operation,
        .offsets = {second_nodes.size() + third_nodes.size() + 1, third_nodes.size() + 1, 1},
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
std::optional<BasicExpression<T>> BasicExpression<T>::from(Unary operation, ExpressionView first, Params params)
    requires std::same_as<BasicExpression, Function>
{
    // add unary operation validation here

    return BasicExpression<T>(operation, first, std::move(params));
}

template <typename T>
std::optional<BasicExpression<T>> BasicExpression<T>::from(
    Binary operation, ExpressionView first, ExpressionView second, Params params)
    requires std::same_as<BasicExpression, Function>
{
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
    Ternary operation, ExpressionView first, ExpressionView second, ExpressionView third, Params params)
    requires std::same_as<BasicExpression, Function>
{
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

template <typename T>
BasicExpression<T>::BasicExpression(const Function& function)
    requires std::same_as<BasicExpression, Expression>
    : nodes(function.nodes) {}

template <typename T>
BasicExpression<T>::BasicExpression(Function&& function) noexcept
    requires std::same_as<BasicExpression, Expression>
    : nodes(std::move(function).nodes) {}

const Tensor& get_tensor(std::span<const Node> nodes) noexcept {
    static_assert(std::variant_size_v<Node> == 2, "assumption below relies on process of elimination");
    // must be Tensor if not FunctionNode
    return *std::get_if<Tensor>(get_subnodes(nodes).data());
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

struct circular_buffers {
    std::size_t inputs;
    std::size_t outputs;
    std::size_t internal;
    std::size_t reusable;

    tt::CBIndex input_id() const noexcept { return tt::CBIndex(inputs + internal - reusable - 1); }
    tt::CBIndex output_id() const noexcept { return tt::CBIndex(inputs + internal); }
    std::size_t total() const noexcept { return inputs + internal + outputs; }
};

auto get_circular_buffers(std::span<const Node> nodes) noexcept {
    circular_buffers result{.outputs = 1};

    for (const auto& node : nodes) {
        if (auto function_ptr = std::get_if<FunctionNode>(&node)) {
            for (const auto offset : function_ptr->offsets) {
                if (std::holds_alternative<FunctionNode>(*(&node - offset))) {
                    // this assumes no common sub-expressions
                    // when CSE is introduced, this will need revised
                    ++result.reusable;
                }
            }

            if (result.reusable > 0) {
                --result.reusable;
            } else {
                ++result.internal;
            }
        } else {
            static_assert(std::variant_size_v<Value> == 2, "assumption below relies on process of elimination");
            ++result.inputs;
        }
    }

    return result;
}

// computes the circular buffer index accounting for re-use between internal nodes
template <typename T>
tt::CBIndex BasicExpressionView<T>::index() const noexcept {
    const auto result = get_circular_buffers(nodes);
    return tt::CBIndex(is_root ? result.output_id() : result.input_id());
}

template <typename T>
tt::CBIndex BasicExpression<T>::index() const noexcept {
    return ExpressionView(*this).index();
}

template <typename T>
std::size_t BasicExpressionView<T>::inputs() const noexcept {
    return get_circular_buffers(nodes).inputs;
}

template <typename T>
std::size_t BasicExpression<T>::inputs() const noexcept {
    return ExpressionView(*this).inputs();
}

template <typename T>
std::size_t BasicExpressionView<T>::circular_buffers() const noexcept {
    return get_circular_buffers(nodes).total();
}

template <typename T>
std::size_t BasicExpression<T>::circular_buffers() const noexcept {
    return ExpressionView(*this).circular_buffers();
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
        return FunctionView(nodes, is_root);
    }

    return std::nullopt;
}

template <typename T>
Value BasicExpressionView<T>::value() const
    requires std::same_as<BasicExpressionView, ExpressionView>
{
    if (const auto tensor_ptr = std::get_if<Tensor>(&root())) {
        return *tensor_ptr;
    }

    static_assert(std::variant_size_v<Value> == 2, "assumption below relies on process of elimination");
    // must be FunctionView if not Tensor
    return FunctionView(nodes, is_root);
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
        result.push_back(ExpressionView(nodes.first(nodes.size() - offset), false));
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
        case RECIP: return "recip";
        case NEGATIVE: return i32_function_name(dtype, "negative");
        case EXP: return "exp";
        case POWER: return "power";
        case EQZ: return i32_u16_u32_function_name(dtype, "eqz");
        case GEZ: return i32_function_name(dtype, "gez");
        case GTZ: return i32_function_name(dtype, "gtz");
        case LEZ: return i32_function_name(dtype, "lez");
        case LTZ: return i32_function_name(dtype, "ltz");
        case NEZ: return i32_u16_u32_function_name(dtype, "nez");
        case LOGICAL_NOT: return i32_u16_u32_function_name(dtype, "logical_not");
    }
}

std::string function_name(DataType dtype, Binary operation) {
    using enum Binary;
    switch (operation) {
        case ADD: return "add";
        case SUB: return "sub";
        case MUL: return "mul";
        case DIV: return "div";
        case POWER: return "power_binary";
    }
}

std::string function_name(DataType dtype, Ternary operation) {
    using enum Ternary;
    switch (operation) {
        case WHERE: return f32_i32_function_name(dtype, "where");
    }
}

void format_to_kernel_string(
    std::back_insert_iterator<std::string> out, DataType dtype, Unary operation, ParamsView params, int& rt_arg_index) {
    switch (params.size()) {
        case 0: fmt::format_to(out, "{}()", function_name(dtype, operation)); return;
        case 1:
            fmt::format_to(out, "{}(get_arg_val<uint32_t>({}))", function_name(dtype, operation), rt_arg_index++);
            return;
        default:
            TT_THROW(
                "unexpected params size {} for unary {} {}",
                params.size(),
                enchantum::to_string(dtype),
                enchantum::to_string(operation));
    }
}

void format_to_kernel_string(
    std::back_insert_iterator<std::string> out,
    DataType dtype,
    Binary operation,
    ParamsView params,
    int& rt_arg_index) {
    switch (params.size()) {
        case 0: fmt::format_to(out, "{}()", function_name(dtype, operation)); return;
        default:
            TT_THROW(
                "unexpected params size {} for binary {} {}",
                params.size(),
                enchantum::to_string(dtype),
                enchantum::to_string(operation));
    }
}

void format_to_kernel_string(
    std::back_insert_iterator<std::string> out,
    DataType dtype,
    Ternary operation,
    ParamsView params,
    int& rt_arg_index) {
    switch (params.size()) {
        case 0: fmt::format_to(out, "{}()", function_name(dtype, operation)); return;
        default:
            TT_THROW(
                "unexpected params size {} for ternary {} {}",
                params.size(),
                enchantum::to_string(dtype),
                enchantum::to_string(operation));
    }
}

void format_to_kernel_string(
    std::back_insert_iterator<std::string> out,
    DataType dtype,
    Operation operation,
    ParamsView params,
    int& rt_arg_index) {
    return std::visit(
        [&](auto alternative) { return lazy::format_to_kernel_string(out, dtype, alternative, params, rt_arg_index); },
        operation);
}

void format_to_kernel_string(std::back_insert_iterator<std::string> out, FunctionView function, int& rt_arg_index) {
    constexpr auto to_cb_id = [](ExpressionView expression) { return enchantum::to_string(expression.index()); };

    fmt::format_to(
        out,
        "with_cb_ids<{},{}>(",
        fmt::join(function.arguments() | std::views::transform(to_cb_id), ","),
        to_cb_id(function));
    lazy::format_to_kernel_string(out, function.dtype(), function.operation(), function.params(), rt_arg_index);
    *out++ = ')';
}

std::optional<Expression> defer(const Tensor& tensor) { return Expression::from(tensor); }

std::optional<Function> defer(Unary unary, ExpressionView first, Params params) {
    return Function::from(unary, first, std::move(params));
}

std::optional<Function> defer(Binary binary, ExpressionView first, ExpressionView second, Params params) {
    return Function::from(binary, first, second, std::move(params));
}

std::optional<Function> defer(
    Ternary ternary, ExpressionView first, ExpressionView second, ExpressionView third, Params params) {
    return Function::from(ternary, first, second, third, std::move(params));
}

std::string to_compute_kernel_string(FunctionView expression) {
    std::string result;
    // n_tiles is index 0
    int rt_arg_index = 1;

    lazy::traverse(
        ttsl::overloaded{
            [](const ttnn::Tensor&) {},
            [&](FunctionView function) {
                // prefix or separator
                if (result.empty()) {
                    std::format_to(
                        std::back_inserter(result), "using View=MakeComputeView<{}>;", expression.circular_buffers());
                    std::format_to(
                        std::back_inserter(result),
                        "View::init_tiles(init_sfpu<c_0,{}>());",
                        enchantum::to_string(expression.index()));
                    result.append("View::compute_tiles(n_tiles,");
                } else {
                    result.push_back(',');
                }

                lazy::format_to_kernel_string(std::back_inserter(result), function, rt_arg_index);
            }},
        expression);

    result.append(");");
    return result;
}

void format_to_debug_string(std::back_insert_iterator<std::string> out, ExpressionView expression);

void format_to_debug_string(std::back_insert_iterator<std::string> out, FunctionView expression) {
    constexpr auto arg_to_string = [](ExpressionView expression) {
        std::string result;
        lazy::format_to_debug_string(std::back_inserter(result), expression);
        return result;
    };
    constexpr auto param_to_string = [](Param param) {
        constexpr auto visitor = [](auto value) { return std::to_string(value); };
        return std::visit(visitor, param);
    };
    const auto type = std::visit(
        ttsl::overloaded{
            [](Unary) { return "Unary"; },
            [](Binary) { return "Binary"; },
            [](Ternary) { return "Ternary"; },
        },
        expression.operation());
    const auto name = std::visit(enchantum::to_string, expression.operation());
    const auto args = expression.arguments() | std::views::transform(arg_to_string);
    const auto params = expression.params() | std::views::transform(param_to_string);
    fmt::format_to(out, "defer({}.{}, {}, [{}])", type, name, fmt::join(args, ", "), fmt::join(params, ", "));
}

void format_to_debug_string(std::back_insert_iterator<std::string> out, ExpressionView expression) {
    if (auto function = expression.function()) {
        return lazy::format_to_debug_string(out, *function);
    }

    fmt::format_to(out, "{}", enchantum::to_string(expression.index()));
}

std::string to_debug_string(FunctionView expression) {
    std::string result;
    lazy::format_to_debug_string(std::back_inserter(result), expression);
    return result;
}

}  // namespace ttnn::operations::lazy
