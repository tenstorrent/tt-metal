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

#include "ttnn/operations/experimental/parallel/device/parallel_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Forward declaration
struct StepDescriptor;

// =============================================================================
// SequentialDeviceOperation Types
// =============================================================================

struct SequentialParams {
    std::vector<std::shared_ptr<StepDescriptor>> steps;
    // Each step carries its own core range
    ttnn::MeshDevice* mesh_device = nullptr;
};

// Empty inputs - actual tensors are in StepDescriptors
struct SequentialInputs {};

// =============================================================================
// StepDescriptor - Abstract interface for type-erased steps
// =============================================================================

struct StepDescriptor {
    virtual ~StepDescriptor() = default;

    StepDescriptor() = default;

    virtual std::vector<const Tensor*> get_input_tensors() const = 0;
    virtual std::vector<TensorSpec> get_output_specs() const = 0;
    virtual std::vector<Tensor> make_output_tensors() const = 0;
    virtual void check_on_cache_hit() const = 0;
    virtual void check_on_cache_miss() const = 0;

    // Get the cores this step should execute on
    virtual const CoreRangeSet& get_cores() const = 0;

    // Add this step's kernels/CBs/semaphores to the program
    // Uses the step's stored core_range to restrict execution to specific cores
    virtual void add_to_program(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) = 0;

    // Override runtime arguments for this step
    virtual void update_runtime_args(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) = 0;

    // Returns whether shared variables have been initialized
    virtual bool has_shared_variables() const = 0;

    // Type information for hashing
    virtual const std::type_info& type_info() const = 0;

    // Get a human-readable name for this step's operation (for profiling)
    virtual std::string operation_name() const = 0;
};

// =============================================================================
// Step<DeviceOp> - User-facing step specification
// =============================================================================

template <typename DeviceOp>
struct Step {
    typename DeviceOp::operation_attributes_t op_attrs;
    typename DeviceOp::tensor_args_t tensor_args;
};

// =============================================================================
// TypedStepDescriptor - Internal implementation
// =============================================================================

template <typename DeviceOp>
struct TypedStepDescriptor : StepDescriptor {
    using operation_attributes_t = typename DeviceOp::operation_attributes_t;
    using tensor_args_t = typename DeviceOp::tensor_args_t;
    using tensor_return_value_t = typename DeviceOp::tensor_return_value_t;
    using program_factory_t = typename DeviceOp::program_factory_t;
    using spec_return_value_t = typename DeviceOp::spec_return_value_t;

    CoreRangeSet cores_;
    operation_attributes_t op_attributes;
    tensor_args_t tensor_args;

    // Type-erased storage for per-step shared variables
    std::any shared_variables_;

    // Index of which variant in program_factory_t was selected
    size_t selected_factory_index_ = 0;

    TypedStepDescriptor(const CoreRangeSet& cores, const Step<DeviceOp>& step) :
        cores_(cores), op_attributes(step.op_attrs), tensor_args(step.tensor_args) {}

    TypedStepDescriptor(const CoreRangeSet& cores, const operation_attributes_t& attrs, const tensor_args_t& args) :
        cores_(cores), op_attributes(attrs), tensor_args(args) {}

    std::vector<const Tensor*> get_input_tensors() const override { return extract_tensors_impl(tensor_args); }

    const CoreRangeSet& get_cores() const override { return cores_; }

    std::vector<TensorSpec> get_output_specs() const override {
        auto specs = compute_specs_for_device_op(op_attributes, tensor_args);
        return flatten_specs(specs);
    }

private:
    static spec_return_value_t compute_specs_for_device_op(
        const operation_attributes_t& attrs, const tensor_args_t& args) {
        return DeviceOp::compute_output_specs(attrs, args);
    }

public:
    std::vector<Tensor> make_output_tensors() const override {
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

                if constexpr (
                    supports_add_to<FactoryType, operation_attributes_t, tensor_args_t, tensor_return_value_t>::value) {
                    // Use the step's stored cores
                    auto shared_vars = factory.add_to(program, op_attributes, tensor_args, tensor_return, cores_);
                    shared_variables_ = std::move(shared_vars);
                } else {
                    TT_FATAL(
                        false,
                        "Factory {} does not support add_to() for sequential composition. "
                        "The operation must implement the add_to() method to be used with ttnn::sequential.",
                        typeid(FactoryType).name());
                }
            },
            factory_variant);

        reflatten_outputs(outputs, tensor_return);
    }

    void update_runtime_args(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) override {
        if (!has_shared_variables()) {
            return;
        }

        auto tensor_return = unflatten_outputs(outputs);
        auto factory_variant = DeviceOp::select_program_factory(op_attributes, tensor_args);

        visit_factory_at_index(factory_variant, selected_factory_index_, [&](auto&& factory) {
            using FactoryType = std::decay_t<decltype(factory)>;
            using SharedVarsType = typename FactoryType::shared_variables_t;

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
                using CachedProgramType = ttnn::device_operation::CachedProgram<SharedVarsType>;
                auto* shared_vars_ptr = std::any_cast<SharedVarsType>(&shared_variables_);
                TT_FATAL(shared_vars_ptr != nullptr, "Shared variables type mismatch");

                CachedProgramType cached_program{std::move(program), std::move(*shared_vars_ptr)};
                factory.override_runtime_arguments(cached_program, op_attributes, tensor_args, tensor_return);

                program = std::move(cached_program.program);
                shared_variables_ = std::move(cached_program.shared_variables);
            }
        });

        reflatten_outputs(outputs, tensor_return);
    }

    bool has_shared_variables() const override { return shared_variables_.has_value(); }

    const std::type_info& type_info() const override { return typeid(DeviceOp); }

    std::string operation_name() const override { return std::string(tt::stl::get_type_name<DeviceOp>()); }

private:
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
// Helper to create StepDescriptor
// =============================================================================

template <typename DeviceOp>
std::shared_ptr<StepDescriptor> create_step(
    const CoreRangeSet& cores,
    const typename DeviceOp::operation_attributes_t& op_attrs,
    const typename DeviceOp::tensor_args_t& tensor_args) {
    return std::make_shared<TypedStepDescriptor<DeviceOp>>(cores, op_attrs, tensor_args);
}

}  // namespace ttnn::experimental::prim

// Convenience aliases for backward compatibility
namespace ttnn::operations::experimental::sequential {
using ttnn::experimental::prim::create_step;
using ttnn::experimental::prim::SequentialParams;
using ttnn::experimental::prim::Step;
using ttnn::experimental::prim::StepDescriptor;
using ttnn::experimental::prim::TypedStepDescriptor;
}  // namespace ttnn::operations::experimental::sequential

// Custom fmt::formatter for StepDescriptor shared_ptr
template <>
struct fmt::formatter<std::shared_ptr<ttnn::experimental::prim::StepDescriptor>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const std::shared_ptr<ttnn::experimental::prim::StepDescriptor>& step, FormatContext& ctx) const {
        if (step) {
            return fmt::format_to(ctx.out(), "{}", step->operation_name());
        }
        return fmt::format_to(ctx.out(), "<null>");
    }
};

// Custom formatter for the steps vector
template <>
struct fmt::formatter<std::vector<std::shared_ptr<ttnn::experimental::prim::StepDescriptor>>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(
        const std::vector<std::shared_ptr<ttnn::experimental::prim::StepDescriptor>>& steps, FormatContext& ctx) const {
        auto out = ctx.out();
        out = fmt::format_to(out, "[");
        for (size_t i = 0; i < steps.size(); ++i) {
            if (i > 0) {
                out = fmt::format_to(out, " -> ");
            }
            if (steps[i]) {
                out = fmt::format_to(out, "{}", steps[i]->operation_name());
            } else {
                out = fmt::format_to(out, "<null>");
            }
        }
        return fmt::format_to(out, "]");
    }
};
