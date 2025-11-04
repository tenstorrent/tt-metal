// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/lazy/lazy_operation.hpp"
#include "ttnn/experimental/lazy/lazy_utils.hpp"
#include "ttnn/device_operation.hpp"
#include <tt_stl/reflection.hpp>
#include <vector>
#include <memory>
#include <tuple>
#include <utility>
#include <boost/pfr.hpp>

namespace ttnn::experimental::lazy {

// Generic wrapper that adapts any device operation to LazyOperation
template <typename operation_t>
    requires ttnn::device_operation::DeviceOperationConcept<operation_t>
class LazyDeviceOperation : public LazyOperation {
public:
    using operation_attributes_t = typename operation_t::operation_attributes_t;
    using tensor_args_t = typename operation_t::tensor_args_t;
    using tensor_return_value_t = typename operation_t::tensor_return_value_t;

    // TODO: It's not greate to capture tensor_args_ in the op
    LazyDeviceOperation(operation_attributes_t attributes, tensor_args_t tensor_args, const std::string& name) :
        attributes_(std::move(attributes)),
        tensor_args_(tensor_args),  // Copy tensor_args (shallow copy since Tensor is shallow-copyable)
        name_(name) {
        // Compute and cache output specs once at construction time
        if constexpr (requires { operation_t::compute_output_specs(attributes_, tensor_args_); }) {
            auto spec = operation_t::compute_output_specs(attributes_, tensor_args_);
            cached_output_specs_ = convert_spec_to_vector(spec);
        }
    }

    // TODO: Can we cache validation result?
    void validate(const std::vector<Tensor>& input_tensors) const {
        TT_FATAL(false, "Not implemented");
        // Call the operation's validation method if it exists
        // Use validate_on_program_cache_miss for comprehensive validation
        // TODO: Should we choose validate_on_program_cache_miss or validate_on_program_cache_hit based on the cache
        // somehow?
        if constexpr (requires { operation_t::validate_on_program_cache_miss(attributes_, tensor_args_); }) {
            operation_t::validate_on_program_cache_miss(attributes_, tensor_args_);
        }
    }

    std::vector<ttnn::TensorSpec> compute_output_specs() const {
        // Return cached output specs that were computed at construction time
        return cached_output_specs_;
    }

    std::string_view name() const override { return std::string_view(name_.c_str()); }

    std::vector<tt::tt_metal::metal_tensor::Tensor> invoke(
        const std::vector<tt::tt_metal::metal_tensor::Tensor>& input_tensors) override {
        // Use the standard device operation eager execution path
        (void)input_tensors;
        auto result = ttnn::device_operation::detail::invoke<operation_t>(attributes_, tensor_args_);

        // Convert result to vector
        return convert_result_to_vector(result);
    }

    const operation_attributes_t& attributes() const { return attributes_; }

    const tensor_args_t& tensor_args() const { return tensor_args_; }

    tt::stl::hash::hash_t operation_type_id() const override { return tt::stl::hash::type_hash<operation_t>; }

    // TODO: Verify vibecoded to_hash
    // Required for hashing support
    tt::stl::hash::hash_t to_hash() const {
        // Hash based on the operation type and attributes
        if constexpr (requires { attributes_.to_hash(); }) {
            return tt::stl::hash::hash_objects_with_default_seed(
                tt::stl::hash::type_hash<operation_t>, attributes_.to_hash());
        } else {
            return tt::stl::hash::hash_objects_with_default_seed(tt::stl::hash::type_hash<operation_t>, attributes_);
        }
    }

private:
    operation_attributes_t attributes_;
    tensor_args_t tensor_args_;
    std::vector<ttnn::TensorSpec> cached_output_specs_;
    std::string name_;
    // Helper to convert any spec type to vector
    template <typename SpecType>
    std::vector<ttnn::TensorSpec> convert_spec_to_vector(const SpecType& spec) const {
        using spec_type = std::decay_t<SpecType>;

        // Handle single TensorSpec
        if constexpr (std::same_as<spec_type, ttnn::TensorSpec>) {
            return {spec};
        }
        // Handle vector<TensorSpec>
        else if constexpr (std::same_as<spec_type, std::vector<ttnn::TensorSpec>>) {
            return spec;
        } else {
            TT_THROW("Unsupported spec type");
        }
    }

    // Convert result from device operation to vector of tensors
    std::vector<tt::tt_metal::metal_tensor::Tensor> convert_result_to_vector(
        const tensor_return_value_t& result) const {
        // TODO: device operations are supposed to return metal tensors, not tnn::Tensor
        if constexpr (std::same_as<tensor_return_value_t, Tensor>) {
            return {result.get_materialized_tensor()};
        } else if constexpr (std::same_as<tensor_return_value_t, std::vector<Tensor>>) {
            std::vector<tt::tt_metal::metal_tensor::Tensor> metal_tensors;
            metal_tensors.reserve(result.size());
            for (const auto& tensor : result) {
                metal_tensors.push_back(tensor.get_materialized_tensor());
            }
            return metal_tensors;
        } else {
            TT_THROW("Unsupported tensor_return_value_t type");
        }
    }
};

// Helper function to create a lazy operation wrapper
template <typename operation_t>
std::shared_ptr<LazyDeviceOperation<operation_t>> make_lazy_device_operation(
    const typename operation_t::operation_attributes_t& attributes,
    const typename operation_t::tensor_args_t& tensor_args,
    const std::string& name) {
    return std::make_shared<LazyDeviceOperation<operation_t>>(attributes, tensor_args, name);
}

}  // namespace ttnn::experimental::lazy
