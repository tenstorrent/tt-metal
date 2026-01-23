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

// =============================================================================
// BranchDescriptor - Value semantics with internal type erasure
// =============================================================================
//
// This class uses the "type erasure" pattern to provide polymorphic behavior
// with value semantics. Instead of exposing an abstract base class and using
// shared_ptr everywhere, BranchDescriptor is a regular value type that can
// be stored in std::vector<BranchDescriptor>.
//
// Copies are cheap (shared ownership via internal shared_ptr), making this
// compatible with frameworks that require copyable types while still
// presenting a clean value-semantics API.
//

class BranchDescriptor {
public:
    // =========================================================================
    // Constructors and assignment
    // =========================================================================

    // Default constructor - creates an empty/invalid descriptor
    BranchDescriptor() = default;

    // Copy and move are all allowed - copies share the underlying data
    BranchDescriptor(BranchDescriptor&&) noexcept = default;
    BranchDescriptor& operator=(BranchDescriptor&&) noexcept = default;
    BranchDescriptor(const BranchDescriptor&) = default;
    BranchDescriptor& operator=(const BranchDescriptor&) = default;

    // =========================================================================
    // Factory method - creates a BranchDescriptor for a specific DeviceOp
    // =========================================================================

    template <typename DeviceOp>
    static BranchDescriptor make(
        const CoreRangeSet& cores,
        const typename DeviceOp::operation_attributes_t& op_attrs,
        const typename DeviceOp::tensor_args_t& tensor_args);

    // =========================================================================
    // Validity check
    // =========================================================================

    explicit operator bool() const { return impl_ != nullptr; }

    // =========================================================================
    // Public interface - delegates to internal implementation
    // =========================================================================

    const CoreRangeSet& core_range() const {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        return impl_->core_range();
    }

    std::vector<const Tensor*> get_input_tensors() const {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        return impl_->get_input_tensors();
    }

    std::vector<TensorSpec> get_output_specs() const {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        return impl_->get_output_specs();
    }

    std::vector<Tensor> make_output_tensors() const {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        return impl_->make_output_tensors();
    }

    void check_on_cache_hit() const {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        impl_->check_on_cache_hit();
    }

    void check_on_cache_miss() const {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        impl_->check_on_cache_miss();
    }

    void add_to_program(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        impl_->add_to_program(program, outputs);
    }

    void update_runtime_args(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        impl_->update_runtime_args(program, outputs);
    }

    bool has_shared_variables() const {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        return impl_->has_shared_variables();
    }

    const std::type_info& type_info() const {
        TT_ASSERT(impl_, "BranchDescriptor is empty");
        return impl_->type_info();
    }

    std::string operation_name() const {
        if (!impl_) {
            return "<empty>";
        }
        return impl_->operation_name();
    }

    // Hash support for tt::stl::hash (via to_hash() method)
    tt::stl::hash::hash_t to_hash() const {
        if (!impl_) {
            return 0;
        }
        return impl_->to_hash();
    }

private:
    // =========================================================================
    // Type Erasure: Concept (abstract interface)
    // =========================================================================

    struct Concept {
        virtual ~Concept() = default;

        virtual const CoreRangeSet& core_range() const = 0;
        virtual std::vector<const Tensor*> get_input_tensors() const = 0;
        virtual std::vector<TensorSpec> get_output_specs() const = 0;
        virtual std::vector<Tensor> make_output_tensors() const = 0;
        virtual void check_on_cache_hit() const = 0;
        virtual void check_on_cache_miss() const = 0;
        virtual void add_to_program(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) = 0;
        virtual void update_runtime_args(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) = 0;
        virtual bool has_shared_variables() const = 0;
        virtual const std::type_info& type_info() const = 0;
        virtual std::string operation_name() const = 0;
        virtual tt::stl::hash::hash_t to_hash() const = 0;
    };

    // =========================================================================
    // Type Erasure: Model<DeviceOp> (concrete implementation)
    // =========================================================================

    template <typename DeviceOp>
    struct Model;

    // Shared ownership allows cheap copies while maintaining value semantics API
    std::shared_ptr<Concept> impl_;
};

// =============================================================================
// ParallelDeviceOperation Types
// =============================================================================

struct ParallelParams {
    std::vector<BranchDescriptor> branches;  // Value semantics with shared ownership internally
    ttnn::MeshDevice* mesh_device = nullptr;
};

// Empty inputs - actual tensors are in BranchDescriptors
struct ParallelInputs {};

// =============================================================================
// Branch<DeviceOp> - User-facing branch specification (for C++ variadic API)
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
// Model<DeviceOp> - Implementation of the type-erased interface
// =============================================================================

template <typename DeviceOp>
struct BranchDescriptor::Model final : BranchDescriptor::Concept {
    using operation_attributes_t = typename DeviceOp::operation_attributes_t;
    using tensor_args_t = typename DeviceOp::tensor_args_t;
    using tensor_return_value_t = typename DeviceOp::tensor_return_value_t;
    using program_factory_t = typename DeviceOp::program_factory_t;
    using spec_return_value_t = typename DeviceOp::spec_return_value_t;

    CoreRangeSet core_range_;
    operation_attributes_t op_attributes_;
    tensor_args_t tensor_args_;

    // Type-erased storage for per-branch shared variables
    std::any shared_variables_;

    // Index of which variant in program_factory_t was selected
    size_t selected_factory_index_ = 0;

    Model(const CoreRangeSet& cores, const operation_attributes_t& attrs, const tensor_args_t& args) :
        core_range_(cores), op_attributes_(attrs), tensor_args_(args) {}

    const CoreRangeSet& core_range() const override { return core_range_; }

    std::vector<const Tensor*> get_input_tensors() const override { return extract_tensors_impl(tensor_args_); }

    std::vector<TensorSpec> get_output_specs() const override {
        auto specs = compute_specs_for_device_op(op_attributes_, tensor_args_);
        return flatten_specs(specs);
    }

    std::vector<Tensor> make_output_tensors() const override {
        auto result = DeviceOp::create_output_tensors(op_attributes_, tensor_args_);
        return flatten_tensors(result);
    }

    void check_on_cache_hit() const override { DeviceOp::validate_on_program_cache_hit(op_attributes_, tensor_args_); }
    void check_on_cache_miss() const override {
        DeviceOp::validate_on_program_cache_miss(op_attributes_, tensor_args_);
    }

    void add_to_program(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) override {
        auto tensor_return = unflatten_outputs(outputs);
        auto factory_variant = DeviceOp::select_program_factory(op_attributes_, tensor_args_);
        selected_factory_index_ = factory_variant.index();

        std::visit(
            [&](auto&& factory) {
                using FactoryType = std::decay_t<decltype(factory)>;

                if constexpr (
                    supports_add_to<FactoryType, operation_attributes_t, tensor_args_t, tensor_return_value_t>::value) {
                    auto shared_vars =
                        factory.add_to(program, op_attributes_, tensor_args_, tensor_return, core_range_);
                    shared_variables_ = std::move(shared_vars);
                } else {
                    TT_FATAL(
                        false,
                        "Factory {} does not support add_to() for parallel composition. "
                        "The operation must implement the add_to() method to be used with ttnn::parallel.",
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
        auto factory_variant = DeviceOp::select_program_factory(op_attributes_, tensor_args_);

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
                    program, *shared_vars_ptr, op_attributes_, tensor_args_, tensor_return);
            } else {
                static_assert(
                    false,
                    "Factory does not support override_runtime_arguments() override needed for "
                    "ttnn::experimental::parallel");
            }
        });

        reflatten_outputs(outputs, tensor_return);
    }

    bool has_shared_variables() const override { return shared_variables_.has_value(); }

    const std::type_info& type_info() const override { return typeid(DeviceOp); }

    std::string operation_name() const override { return std::string(tt::stl::get_type_name<DeviceOp>()); }

    tt::stl::hash::hash_t to_hash() const override {
        // Hash the operation type and core range
        tt::stl::hash::hash_t h = typeid(DeviceOp).hash_code();
        for (const auto& range : core_range_.ranges()) {
            h = tt::stl::hash::hash_objects(
                h, range.start_coord.x, range.start_coord.y, range.end_coord.x, range.end_coord.y);
        }
        return h;
    }

private:
    static spec_return_value_t compute_specs_for_device_op(
        const operation_attributes_t& attrs, const tensor_args_t& args) {
        return DeviceOp::compute_output_specs(attrs, args);
    }

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
// BranchDescriptor::make<DeviceOp>() implementation
// =============================================================================

template <typename DeviceOp>
BranchDescriptor BranchDescriptor::make(
    const CoreRangeSet& cores,
    const typename DeviceOp::operation_attributes_t& op_attrs,
    const typename DeviceOp::tensor_args_t& tensor_args) {
    BranchDescriptor result;
    result.impl_ = std::make_shared<Model<DeviceOp>>(cores, op_attrs, tensor_args);
    return result;
}

// =============================================================================
// Helper to convert Branch<DeviceOp> to BranchDescriptor
// =============================================================================

template <typename DeviceOp>
BranchDescriptor make_descriptor(const Branch<DeviceOp>& branch) {
    return BranchDescriptor::make<DeviceOp>(branch.cores, branch.op_attrs, branch.tensor_args);
}

// =============================================================================
// create_branch - Convenient helper for operations to create branch descriptors
// =============================================================================
//
// Usage in an operation's branch() method:
//
//   static BranchDescriptor branch(
//       const Tensor& input, float epsilon, const CoreRangeSet& cores, ...) {
//       return ttnn::experimental::prim::create_branch<LayerNormDeviceOperation>(
//           cores,
//           operation_attributes_t{.eps = epsilon, ...},
//           tensor_args_t{.input = input, ...}
//       );
//   }
//
template <typename DeviceOp>
BranchDescriptor create_branch(
    const CoreRangeSet& cores,
    const typename DeviceOp::operation_attributes_t& op_attrs,
    const typename DeviceOp::tensor_args_t& tensor_args) {
    return BranchDescriptor::make<DeviceOp>(cores, op_attrs, tensor_args);
}

}  // namespace ttnn::experimental::prim

// Convenience aliases for backward compatibility and easier access
namespace ttnn::operations::experimental {
using ttnn::experimental::prim::Branch;
using ttnn::experimental::prim::BranchDescriptor;
using ttnn::experimental::prim::create_branch;
using ttnn::experimental::prim::make_descriptor;
using ttnn::experimental::prim::ParallelParams;
using ttnn::experimental::prim::supports_add_to;
}  // namespace ttnn::operations::experimental

// Additional convenience alias for easier access
namespace ttnn::parallel_internal {
using ttnn::experimental::prim::BranchDescriptor;
using ttnn::experimental::prim::create_branch;
}  // namespace ttnn::parallel_internal

// Custom fmt::formatter for BranchDescriptor to show operation names in profiler
template <>
struct fmt::formatter<ttnn::experimental::prim::BranchDescriptor> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const ttnn::experimental::prim::BranchDescriptor& branch, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}", branch.operation_name());
    }
};

// Custom formatter for the branches vector to show as a list of operation names
template <>
struct fmt::formatter<std::vector<ttnn::experimental::prim::BranchDescriptor>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const std::vector<ttnn::experimental::prim::BranchDescriptor>& branches, FormatContext& ctx) const {
        auto out = ctx.out();
        out = fmt::format_to(out, "[");
        for (size_t i = 0; i < branches.size(); ++i) {
            if (i > 0) {
                out = fmt::format_to(out, ", ");
            }
            out = fmt::format_to(out, "{}", branches[i].operation_name());
        }
        return fmt::format_to(out, "]");
    }
};
