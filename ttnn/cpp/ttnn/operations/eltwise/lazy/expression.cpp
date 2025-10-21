// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
BasicExpression<T>::BasicExpression(Unary operation, ExpressionView first)
    requires std::same_as<BasicExpression, Function>
{
    const auto first_nodes = get_subnodes(first.nodes);
    nodes.reserve(first_nodes.size() + 1);
    nodes.insert(nodes.end(), first_nodes.begin(), first_nodes.end());
    nodes.emplace_back(FunctionNode{.operation = operation, .offsets = {1}});
}

template <typename T>
BasicExpression<T>::BasicExpression(UnaryWithParam operation, ExpressionView first, Param second)
    requires std::same_as<BasicExpression, Function>
{
    const auto first_nodes = get_subnodes(first.nodes);
    nodes.reserve(first_nodes.size() + 1);
    nodes.insert(nodes.end(), first_nodes.begin(), first_nodes.end());
    nodes.emplace_back(FunctionNode{.operation = operation, .offsets = {1}, .params = {second}});
}

template <typename T>
BasicExpression<T>::BasicExpression(Binary operation, ExpressionView first, ExpressionView second)
    requires std::same_as<BasicExpression, Function>
{
    const auto first_nodes = get_subnodes(first.nodes);
    const auto second_nodes = get_subnodes(second.nodes);
    nodes.reserve(first_nodes.size() + second_nodes.size() + 1);
    nodes.insert(nodes.end(), first_nodes.begin(), first_nodes.end());
    nodes.insert(nodes.end(), second_nodes.begin(), second_nodes.end());
    nodes.emplace_back(FunctionNode{.operation = operation, .offsets = {second_nodes.size() + 1, 1}});
}

template <typename T>
BasicExpression<T>::BasicExpression(
    Ternary operation, ExpressionView first, ExpressionView second, ExpressionView third)
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
        .operation = operation, .offsets = {second_nodes.size() + third_nodes.size() + 1, third_nodes.size() + 1, 1}});
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

    // TODO relax dtype and memory_config requirements
    if (tensor.dtype() != DataType::BFLOAT16) {
        return std::nullopt;
    }

    if (tensor.memory_config().is_sharded()) {
        return std::nullopt;
    }

    // add tensor validation here

    return BasicExpression<T>(tensor);
}

template <typename T>
std::optional<BasicExpression<T>> BasicExpression<T>::from(Unary operation, ExpressionView first)
    requires std::same_as<BasicExpression, Function>
{
    // add unary validation here

    return BasicExpression<T>(operation, first);
}

template <typename T>
std::optional<BasicExpression<T>> BasicExpression<T>::from(UnaryWithParam operation, ExpressionView first, Param second)
    requires std::same_as<BasicExpression, Function>
{
    // add unary scalar validation here

    return BasicExpression<T>(operation, first, second);
}

template <typename T>
std::optional<BasicExpression<T>> BasicExpression<T>::from(
    Binary operation, ExpressionView first, ExpressionView second)
    requires std::same_as<BasicExpression, Function>
{
    if (first.dtype() != second.dtype()) {
        return std::nullopt;
    }

    if (first.logical_shape() != second.logical_shape()) {
        return std::nullopt;
    }

    // add binary validation here

    auto expression = BasicExpression<T>(operation, first, second);

    if (expression.cb_index() >= tt::CBIndex::SIZE) {
        return std::nullopt;
    }

    return expression;
}

template <typename T>
std::optional<BasicExpression<T>> BasicExpression<T>::from(
    Ternary operation, ExpressionView first, ExpressionView second, ExpressionView third)
    requires std::same_as<BasicExpression, Function>
{
    if (first.dtype() != second.dtype() or first.dtype() != third.dtype()) {
        return std::nullopt;
    }

    if (first.logical_shape() != second.logical_shape() or first.logical_shape() != third.logical_shape()) {
        return std::nullopt;
    }

    // add ternary validation here

    auto expression = BasicExpression<T>(operation, first, second, third);

    if (expression.cb_index() >= tt::CBIndex::SIZE) {
        return std::nullopt;
    }

    return expression;
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
    std::size_t internal;
    std::size_t outputs;
    tt::CBIndex internal_id;

    tt::CBIndex input_id() const noexcept { return tt::CBIndex(inputs + internal - 1); }
    tt::CBIndex output_id() const noexcept { return tt::CBIndex(inputs + internal); }
    std::size_t total() const noexcept { return inputs + internal + outputs; }
};

auto get_circular_buffers(std::span<const Node> nodes) noexcept {
    const auto nodes_ptr = nodes.data();
    circular_buffers result{.outputs = 1};
    std::unordered_map<std::size_t, tt::CBIndex> internal_ids;
    std::queue<tt::CBIndex> reusable_ids;

    for (const auto& parent : nodes) {
        if (auto function_ptr = std::get_if<FunctionNode>(&parent)) {
            for (const auto offset : function_ptr->offsets) {
                if (const auto child_ptr = &parent - offset; std::holds_alternative<FunctionNode>(*child_ptr)) {
                    // this assumes no common subexpressions
                    // TODO apply CSE optimization
                    reusable_ids.push(internal_ids.find(child_ptr - nodes_ptr)->second);
                }
            }
            // if no reusable nodes available
            if (reusable_ids.empty()) {
                // new internal id is next output id
                result.internal_id = result.output_id();
                ++result.internal;
            } else {
                // otherwise new internal id is first reusable id
                result.internal_id = reusable_ids.front();
                reusable_ids.pop();
            }

            internal_ids.emplace(&parent - nodes_ptr, result.internal_id);
        } else {
            static_assert(std::variant_size_v<Value> == 2, "assumption below relies on process of elimination");
            ++result.inputs;
        }
    }

    return result;
}

auto get_runtime_arguments(std::span<const Node> nodes) noexcept {
    std::size_t runtime_arguments = 0;

    for (const auto& node : nodes) {
        if (auto function_ptr = std::get_if<FunctionNode>(&node)) {
            runtime_arguments += function_ptr->params.size();
        }
    }

    return runtime_arguments;
}

// computes the circular buffer index accounting for re-use between internal nodes
template <typename T>
tt::CBIndex BasicExpressionView<T>::cb_index() const noexcept {
    const auto result = get_circular_buffers(nodes);
    const auto is_function = std::holds_alternative<FunctionNode>(root());
    return tt::CBIndex(is_root ? result.output_id() : is_function ? result.internal_id : result.input_id());
}

template <typename T>
tt::CBIndex BasicExpression<T>::cb_index() const noexcept {
    return ExpressionView(*this).cb_index();
}

// computes runtime argument offset
template <typename T>
std::size_t BasicExpressionView<T>::rt_offset() const noexcept {
    return get_runtime_arguments(nodes.first(nodes.size() - 1));
}

template <typename T>
std::size_t BasicExpression<T>::rt_offset() const noexcept {
    return ExpressionView(*this).rt_offset();
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

// TODO use std::optional<std::string> return type and re-use for validation above

std::string function_name_i32_u16_u32(DataType dtype, std::string operation) {
    using enum DataType;
    switch (dtype) {
        case INT32: return fmt::format("{}_int32", operation);
        case UINT16: return fmt::format("{}_uint16", operation);
        case UINT32: return fmt::format("{}_uint32", operation);
        default: return operation;
    }
}

std::string function_name_i32(DataType dtype, std::string operation) {
    using enum DataType;
    switch (dtype) {
        case INT32: return fmt::format("{}_int32", operation);
        default: return operation;
    }
}

std::string function_name_f32_i32(DataType dtype, std::string operation) {
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
        case RECIP: return "recip";
        case NEGATIVE: return lazy::function_name_i32(dtype, "negative");
        case EXP: return "exp";
        case EQZ: return lazy::function_name_i32_u16_u32(dtype, "eqz");
        case GEZ: return lazy::function_name_i32(dtype, "gez");
        case GTZ: return lazy::function_name_i32(dtype, "gtz");
        case LEZ: return lazy::function_name_i32(dtype, "lez");
        case LTZ: return lazy::function_name_i32(dtype, "ltz");
        case NEZ: return lazy::function_name_i32_u16_u32(dtype, "nez");
        case LOGICAL_NOT: return lazy::function_name_i32_u16_u32(dtype, "logical_not");
        case ATAN: return "atan";
    }
}

std::string function_name(DataType dtype, UnaryWithParam operation) {
    using enum UnaryWithParam;
    switch (operation) {
        case ADD: return lazy::function_name_i32(dtype, "add_unary");
        case SUB: return "sub_unary";
        case RSUB: return "rsub_unary";
        case MUL: return "mul_unary";
        case DIV: return "div_unary";
        case POWER: return "power";
        case RPOW: return "rpow";
    }
}

std::string function_name(DataType dtype, Binary operation) {
    using enum Binary;
    switch (operation) {
        case ADD: return "add";
        case SUB: return "sub";
        case MUL: return "mul";
        case DIV: return "div_binary";
        case POWER: return "power_binary";
    }
}

std::string function_name(DataType dtype, Ternary operation) {
    using enum Ternary;
    switch (operation) {
        case WHERE: return lazy::function_name_f32_i32(dtype, "where");
    }
}

void format_to_kernel_string(
    std::back_insert_iterator<std::string> out, DataType dtype, Operation operation, std::size_t rt_offset) {
    return std::visit(
        ttsl::overloaded{
            [&](UnaryWithParam alternative) {
                fmt::format_to(
                    out, "{}(get_arg_val<uint32_t>({}))", lazy::function_name(dtype, alternative), rt_offset);
            },
            [&](auto alternative) { fmt::format_to(out, "{}()", lazy::function_name(dtype, alternative)); }},
        operation);
}

void format_to_kernel_string(std::back_insert_iterator<std::string> out, FunctionView function) {
    constexpr auto to_cb_index = [](ExpressionView expression) { return enchantum::to_string(expression.cb_index()); };

    fmt::format_to(
        out,
        "with_cb_ids<{},{}>(",
        fmt::join(function.arguments() | std::views::transform(to_cb_index), ","),
        to_cb_index(function));
    // n_tiles precedes runtime arguments in expression tree
    lazy::format_to_kernel_string(out, function.dtype(), function.operation(), function.rt_offset() + 1);
    *out++ = ')';
}

std::optional<Expression> defer(const Tensor& tensor) { return Expression::from(tensor); }

std::optional<Function> defer(Unary operation, ExpressionView first) { return Function::from(operation, first); }

std::optional<Function> defer(UnaryWithParam operation, ExpressionView first, Param second) {
    return Function::from(operation, first, second);
}

std::optional<Function> defer(Binary operation, ExpressionView first, ExpressionView second) {
    return Function::from(operation, first, second);
}

std::optional<Function> defer(Ternary operation, ExpressionView first, ExpressionView second, ExpressionView third) {
    return Function::from(operation, first, second, third);
}

std::string to_compute_kernel_string(FunctionView expression) {
    std::string result;

    lazy::traverse(
        ttsl::overloaded{
            [](const ttnn::Tensor&) {},
            [&](FunctionView function) {
                // prefix or separator
                if (result.empty()) {
                    fmt::format_to(
                        std::back_inserter(result), "using View=MakeComputeView<{}>;", expression.circular_buffers());
                    fmt::format_to(
                        std::back_inserter(result),
                        "View::init_tiles(init_sfpu<c_0,{}>());",
                        enchantum::to_string(expression.cb_index()));
                    result.append("View::compute_tiles<num_tiles_per_cycle>(n_tiles,");
                } else {
                    result.push_back(',');
                }

                lazy::format_to_kernel_string(std::back_inserter(result), function);
            }},
        expression);

    result.append(")");
    return result;
}

void format_to_debug_string(std::back_insert_iterator<std::string> out, ExpressionView expression);

void format_to_debug_string(std::back_insert_iterator<std::string> out, FunctionView expression) {
    constexpr auto to_lower = [](unsigned char ch) -> char { return std::tolower(ch); };
    constexpr auto arg_to_string = [](ExpressionView expression) {
        std::string result;
        lazy::format_to_debug_string(std::back_inserter(result), expression);
        return result;
    };
    constexpr auto param_to_string = [](Param param) {
        constexpr auto visitor = [](auto value) { return std::to_string(value); };
        return std::visit(visitor, param);
    };
    const auto name = std::visit(enchantum::to_string, expression.operation()) | std::views::transform(to_lower);
    const auto args = expression.arguments() | std::views::transform(arg_to_string);
    const auto params = expression.params() | std::views::transform(param_to_string);
    if (params.empty()) {
        fmt::format_to(out, "{}({})", fmt::join(name, ""), fmt::join(args, ", "));
    } else {
        fmt::format_to(out, "{}({}, {})", fmt::join(name, ""), fmt::join(args, ", "), fmt::join(params, ", "));
    }
}

void format_to_debug_string(std::back_insert_iterator<std::string> out, ExpressionView expression) {
    if (auto function = expression.function()) {
        return lazy::format_to_debug_string(out, *function);
    }

    fmt::format_to(out, "{}", enchantum::to_string(expression.cb_index()));
}

std::string to_debug_string(FunctionView expression) {
    std::string result;
    lazy::format_to_debug_string(std::back_inserter(result), expression);
    return result;
}

}  // namespace ttnn::operations::lazy
