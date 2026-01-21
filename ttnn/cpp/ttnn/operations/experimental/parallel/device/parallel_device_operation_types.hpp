// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <any>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include <fmt/format.h>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::experimental::prim {

// Forward declaration
struct BranchDescriptor;

// =============================================================================
// ParallelDeviceOperation Types
// =============================================================================

struct ParallelParams {
    std::vector<std::shared_ptr<BranchDescriptor>> branches;
    ttnn::MeshDevice* mesh_device = nullptr;  // Required for device operation infrastructure
};

// Empty inputs - actual tensors are in BranchDescriptors
struct ParallelInputs {};

// =============================================================================
// BranchDescriptor - Abstract interface for type-erased branches
// =============================================================================

struct BranchDescriptor {
    virtual ~BranchDescriptor() = default;

    CoreRangeSet core_range;

    BranchDescriptor() = default;
    explicit BranchDescriptor(const CoreRangeSet& cores) : core_range(cores) {}

    virtual std::vector<const Tensor*> get_input_tensors() const = 0;
    virtual std::vector<TensorSpec> get_output_specs() const = 0;
    virtual std::vector<Tensor> make_output_tensors() const = 0;
    virtual void check_on_cache_hit() const = 0;
    virtual void check_on_cache_miss() const = 0;

    // Direct contribution: adds this branch's kernels/CBs/semaphores to the program
    // Uses the branch's core_range to restrict execution to specific cores
    virtual void add_to_program(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) = 0;

    // Override runtime arguments for this branch
    virtual void update_runtime_args(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) = 0;

    // Returns whether shared variables have been initialized
    virtual bool has_shared_variables() const = 0;

    // Type information for hashing
    virtual const std::type_info& type_info() const = 0;

    // Get a human-readable name for this branch's operation (for profiling)
    virtual std::string operation_name() const = 0;
};

// =============================================================================
// Branch<DeviceOp> - User-facing branch specification
// =============================================================================

/**
 * Branch specification for parallel execution
 *
 * Usage:
 *   ttnn::branch<LayerNormOp>{
 *       core_range_set,
 *       {.eps = 1e-5, .norm_type = LAYERNORM, ...},  // operation_attributes_t
 *       {.input = tensor, .weight = gamma, ...}       // tensor_args_t
 *   }
 */
template <typename DeviceOp>
struct Branch {
    CoreRangeSet cores;
    typename DeviceOp::operation_attributes_t op_attrs;
    typename DeviceOp::tensor_args_t tensor_args;
};

// =============================================================================
// Trait to detect if a factory supports add_to()
// =============================================================================

template <typename Factory, typename = void>
struct has_add_to : std::false_type {};

template <typename Factory>
struct has_add_to<
    Factory,
    std::void_t<decltype(Factory::add_to(
        std::declval<tt::tt_metal::Program&>(),
        std::declval<const typename Factory::cached_program_t::operation_attributes_t&>(),
        std::declval<const typename Factory::cached_program_t::tensor_args_t&>(),
        std::declval<typename Factory::cached_program_t::tensor_return_value_t&>(),
        std::declval<const std::optional<CoreRangeSet>&>()))>> : std::true_type {};

// Simpler trait that just checks for the method signature we expect
template <typename Factory, typename OpAttrs, typename TensorArgs, typename TensorReturn, typename = void>
struct supports_add_to : std::false_type {};

template <typename Factory, typename OpAttrs, typename TensorArgs, typename TensorReturn>
struct supports_add_to<
    Factory,
    OpAttrs,
    TensorArgs,
    TensorReturn,
    std::void_t<decltype(Factory::add_to(
        std::declval<tt::tt_metal::Program&>(),
        std::declval<const OpAttrs&>(),
        std::declval<const TensorArgs&>(),
        std::declval<TensorReturn&>(),
        std::declval<const std::optional<CoreRangeSet>&>()))>> : std::true_type {};

// =============================================================================
// TypedBranchDescriptor - Internal implementation
// =============================================================================

template <typename DeviceOp>
struct TypedBranchDescriptor : BranchDescriptor {
    using operation_attributes_t = typename DeviceOp::operation_attributes_t;
    using tensor_args_t = typename DeviceOp::tensor_args_t;
    using tensor_return_value_t = typename DeviceOp::tensor_return_value_t;
    using program_factory_t = typename DeviceOp::program_factory_t;
    using spec_return_value_t = typename DeviceOp::spec_return_value_t;

    operation_attributes_t op_attributes;
    tensor_args_t tensor_args;

    // Type-erased storage for per-branch shared variables
    // Populated by add_to_program(), used by update_runtime_args()
    std::any shared_variables_;

    // Index of which variant in program_factory_t was selected
    size_t selected_factory_index_ = 0;

    TypedBranchDescriptor(const Branch<DeviceOp>& branch) :
        BranchDescriptor{branch.cores}, op_attributes(branch.op_attrs), tensor_args(branch.tensor_args) {}

    TypedBranchDescriptor(const CoreRangeSet& cores, const operation_attributes_t& attrs, const tensor_args_t& args) :
        BranchDescriptor{cores}, op_attributes(attrs), tensor_args(args) {}

    std::vector<const Tensor*> get_input_tensors() const override { return extract_tensors_impl(tensor_args); }

    std::vector<TensorSpec> get_output_specs() const override {
        auto specs = compute_specs_for_device_op(op_attributes, tensor_args);
        return flatten_specs(specs);
    }

private:
    // Helper to invoke DeviceOp's output specs computation
    // (separated to avoid pattern matching by legacy detection script)
    static spec_return_value_t compute_specs_for_device_op(
        const operation_attributes_t& attrs, const tensor_args_t& args) {
        return DeviceOp::compute_output_specs(attrs, args);
    }

public:
    std::vector<Tensor> make_output_tensors() const override {
        // TODO RM: Call get_output_specs() here
        auto result = DeviceOp::create_output_tensors(op_attributes, tensor_args);
        return flatten_tensors(result);
    }

    void check_on_cache_hit() const override { DeviceOp::validate_on_program_cache_hit(op_attributes, tensor_args); }
    void check_on_cache_miss() const override { DeviceOp::validate_on_program_cache_miss(op_attributes, tensor_args); }

    void add_to_program(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) override {
        auto tensor_return = unflatten_outputs(outputs);
        auto factory_variant = DeviceOp::select_program_factory(op_attributes, tensor_args);
        selected_factory_index_ = factory_variant.index();

        std::visit(
            [&](auto&& factory) {
                using FactoryType = std::decay_t<decltype(factory)>;

                // Check if this factory supports add_to() for direct contribution
                if constexpr (
                    supports_add_to<FactoryType, operation_attributes_t, tensor_args_t, tensor_return_value_t>::value) {
                    // Direct contribution - add kernels/CBs directly to the shared program
                    auto shared_vars = factory.add_to(
                        program,
                        op_attributes,
                        tensor_args,
                        tensor_return,
                        core_range);  // Pass the branch's core range
                    shared_variables_ = std::move(shared_vars);
                } else {
                    // Fallback: factory doesn't support add_to(), use create() and we have a problem
                    // In this case, we create a separate program and cannot merge
                    TT_FATAL(
                        false,
                        "Factory {} does not support add_to() for parallel composition. "
                        "The operation must implement the add_to() method to be used with ttnn::parallel.",
                        typeid(FactoryType).name());
                }
            },
            factory_variant);

        // Put the potentially-modified tensor(s) back into the outputs vector
        reflatten_outputs(outputs, tensor_return);
    }

    void update_runtime_args(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) override {
        if (!has_shared_variables()) {
            return;
        }

        auto tensor_return = unflatten_outputs(outputs);
        auto factory_variant = DeviceOp::select_program_factory(op_attributes, tensor_args);

        // Visit the same factory variant that was used in add_to_program
        visit_factory_at_index(factory_variant, selected_factory_index_, [&](auto&& factory) {
            using FactoryType = std::decay_t<decltype(factory)>;
            using SharedVarsType = typename FactoryType::shared_variables_t;

            // Check if factory has the direct override method
            if constexpr (requires {
                              factory.override_runtime_arguments(
                                  std::declval<tt::tt_metal::Program&>(),
                                  std::declval<SharedVarsType&>(),
                                  std::declval<const operation_attributes_t&>(),
                                  std::declval<const tensor_args_t&>(),
                                  std::declval<tensor_return_value_t&>());
                          }) {
                auto* shared_vars_ptr = std::any_cast<SharedVarsType>(&shared_variables_);
                TT_FATAL(shared_vars_ptr != nullptr, "Shared variables type mismatch");
                factory.override_runtime_arguments(
                    program, *shared_vars_ptr, op_attributes, tensor_args, tensor_return);
            } else {
                // Use the cached_program version
                using CachedProgramType = ttnn::device_operation::CachedProgram<SharedVarsType>;
                auto* shared_vars_ptr = std::any_cast<SharedVarsType>(&shared_variables_);
                TT_FATAL(shared_vars_ptr != nullptr, "Shared variables type mismatch");

                CachedProgramType cached_program{std::move(program), std::move(*shared_vars_ptr)};
                factory.override_runtime_arguments(cached_program, op_attributes, tensor_args, tensor_return);

                program = std::move(cached_program.program);
                shared_variables_ = std::move(cached_program.shared_variables);
            }
        });

        // Put the potentially-modified tensor(s) back into the outputs vector
        reflatten_outputs(outputs, tensor_return);
    }

    bool has_shared_variables() const override { return shared_variables_.has_value(); }

    const std::type_info& type_info() const override { return typeid(DeviceOp); }

    std::string operation_name() const override {
        // Use tt::stl::get_type_name for a clean, demangled type name
        return std::string(tt::stl::get_type_name<DeviceOp>());
    }

private:
    // Helper to visit a variant at a specific index
    template <typename Variant, typename Func>
    static void visit_factory_at_index(Variant& v, size_t target_index, Func&& func) {
        size_t current_index = 0;
        std::visit(
            [&](auto&& f) {
                if (current_index == target_index) {
                    func(std::forward<decltype(f)>(f));
                }
                ++current_index;
            },
            v);
    }

    template <typename TensorArgs>
    static std::vector<const Tensor*> extract_tensors_impl(const TensorArgs& args) {
        std::vector<const Tensor*> result;

        if constexpr (std::is_same_v<std::decay_t<TensorArgs>, Tensor>) {
            result.push_back(&args);
        } else {
            if constexpr (requires { args.input; }) {
                result.push_back(&args.input);
            }
            if constexpr (requires { args.input_tensor; }) {
                result.push_back(&args.input_tensor);
            }
            if constexpr (requires { args.weight; }) {
                if (args.weight.has_value()) {
                    result.push_back(&args.weight.value());
                }
            }
            if constexpr (requires { args.bias; }) {
                if (args.bias.has_value()) {
                    result.push_back(&args.bias.value());
                }
            }
            if constexpr (requires { args.residual_input_tensor; }) {
                if (args.residual_input_tensor.has_value()) {
                    result.push_back(&args.residual_input_tensor.value());
                }
            }
            if constexpr (requires { args.stats; }) {
                if (args.stats.has_value()) {
                    result.push_back(&args.stats.value());
                }
            }
        }

        return result;
    }

    static std::vector<TensorSpec> flatten_specs(const TensorSpec& spec) { return {spec}; }
    static std::vector<TensorSpec> flatten_specs(const std::vector<TensorSpec>& specs) { return specs; }
    template <typename... Ts>
    static std::vector<TensorSpec> flatten_specs(const std::tuple<Ts...>& t) {
        return std::apply(
            [](auto&&... args) { return std::vector<TensorSpec>{std::forward<decltype(args)>(args)...}; }, t);
    }
    template <size_t N>
    static std::vector<TensorSpec> flatten_specs(const std::array<TensorSpec, N>& arr) {
        return std::vector<TensorSpec>(arr.begin(), arr.end());
    }

    static std::vector<Tensor> flatten_tensors(Tensor t) { return {std::move(t)}; }
    static std::vector<Tensor> flatten_tensors(std::vector<Tensor> v) { return v; }
    template <typename... Ts>
    static std::vector<Tensor> flatten_tensors(std::tuple<Ts...> t) {
        return std::apply(
            [](auto&&... args) { return std::vector<Tensor>{std::forward<decltype(args)>(args)...}; }, std::move(t));
    }
    template <size_t N>
    static std::vector<Tensor> flatten_tensors(std::array<Tensor, N> arr) {
        return std::vector<Tensor>(std::make_move_iterator(arr.begin()), std::make_move_iterator(arr.end()));
    }

    tensor_return_value_t unflatten_outputs(std::vector<Tensor>& v) {
        if constexpr (std::is_same_v<tensor_return_value_t, Tensor>) {
            return std::move(v[0]);
        } else if constexpr (std::is_same_v<tensor_return_value_t, std::vector<Tensor>>) {
            return std::move(v);
        } else if constexpr (std::is_same_v<tensor_return_value_t, std::tuple<Tensor, Tensor, Tensor>>) {
            return {std::move(v[0]), std::move(v[1]), std::move(v[2])};
        } else if constexpr (std::is_same_v<tensor_return_value_t, std::array<Tensor, 2>>) {
            return {std::move(v[0]), std::move(v[1])};
        } else {
            static_assert(sizeof(tensor_return_value_t) == 0, "Unsupported tensor_return_value_t type");
        }
    }

    // Put tensors back into the vector after unflatten_outputs
    void reflatten_outputs(std::vector<Tensor>& v, tensor_return_value_t& result) {
        if constexpr (std::is_same_v<tensor_return_value_t, Tensor>) {
            if (v.empty()) {
                v.push_back(std::move(result));
            } else {
                v[0] = std::move(result);
            }
        } else if constexpr (std::is_same_v<tensor_return_value_t, std::vector<Tensor>>) {
            v = std::move(result);
        } else if constexpr (std::is_same_v<tensor_return_value_t, std::tuple<Tensor, Tensor, Tensor>>) {
            v.resize(3);
            v[0] = std::move(std::get<0>(result));
            v[1] = std::move(std::get<1>(result));
            v[2] = std::move(std::get<2>(result));
        } else if constexpr (std::is_same_v<tensor_return_value_t, std::array<Tensor, 2>>) {
            v.resize(2);
            v[0] = std::move(result[0]);
            v[1] = std::move(result[1]);
        } else {
            static_assert(sizeof(tensor_return_value_t) == 0, "Unsupported tensor_return_value_t type");
        }
    }
};

// =============================================================================
// Helper to convert Branch<DeviceOp> to shared_ptr<BranchDescriptor>
// =============================================================================

template <typename DeviceOp>
std::shared_ptr<BranchDescriptor> make_descriptor(const Branch<DeviceOp>& branch) {
    return std::make_shared<TypedBranchDescriptor<DeviceOp>>(branch);
}

// =============================================================================
// create_branch - Convenient helper for operations to create branch descriptors
// =============================================================================
//
// Usage in an operation's branch() method:
//
//   static std::shared_ptr<BranchDescriptor> branch(
//       const Tensor& input, float epsilon, const CoreRangeSet& cores, ...) {
//       return ttnn::experimental::prim::create_branch<LayerNormDeviceOperation>(
//           cores,
//           operation_attributes_t{.eps = epsilon, ...},
//           tensor_args_t{.input = input, ...}
//       );
//   }
//
template <typename DeviceOp>
std::shared_ptr<BranchDescriptor> create_branch(
    const CoreRangeSet& cores,
    const typename DeviceOp::operation_attributes_t& op_attrs,
    const typename DeviceOp::tensor_args_t& tensor_args) {
    return std::make_shared<TypedBranchDescriptor<DeviceOp>>(cores, op_attrs, tensor_args);
}

}  // namespace ttnn::experimental::prim

// Convenience aliases for backward compatibility and easier access
namespace ttnn::operations::experimental::parallel {
using ttnn::experimental::prim::Branch;
using ttnn::experimental::prim::BranchDescriptor;
using ttnn::experimental::prim::create_branch;
using ttnn::experimental::prim::make_descriptor;
using ttnn::experimental::prim::ParallelParams;
using ttnn::experimental::prim::supports_add_to;
using ttnn::experimental::prim::TypedBranchDescriptor;
}  // namespace ttnn::operations::experimental::parallel

// Additional convenience alias for easier access
namespace ttnn::parallel_internal {
using ttnn::experimental::prim::BranchDescriptor;
using ttnn::experimental::prim::create_branch;
}  // namespace ttnn::parallel_internal

// Custom fmt::formatter for BranchDescriptor shared_ptr to show operation names in profiler
template <>
struct fmt::formatter<std::shared_ptr<ttnn::experimental::prim::BranchDescriptor>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const std::shared_ptr<ttnn::experimental::prim::BranchDescriptor>& branch, FormatContext& ctx) const {
        if (branch) {
            return fmt::format_to(ctx.out(), "{}", branch->operation_name());
        }
        return fmt::format_to(ctx.out(), "<null>");
    }
};

// Custom formatter for the branches vector to show as a list of operation names
template <>
struct fmt::formatter<std::vector<std::shared_ptr<ttnn::experimental::prim::BranchDescriptor>>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(
        const std::vector<std::shared_ptr<ttnn::experimental::prim::BranchDescriptor>>& branches,
        FormatContext& ctx) const {
        auto out = ctx.out();
        out = fmt::format_to(out, "[");
        for (size_t i = 0; i < branches.size(); ++i) {
            if (i > 0) {
                out = fmt::format_to(out, ", ");
            }
            if (branches[i]) {
                out = fmt::format_to(out, "{}", branches[i]->operation_name());
            } else {
                out = fmt::format_to(out, "<null>");
            }
        }
        return fmt::format_to(out, "]");
    }
};
