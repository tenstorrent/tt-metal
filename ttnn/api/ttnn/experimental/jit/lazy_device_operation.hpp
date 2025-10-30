// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/jit/lazy_operation.hpp"
#include "ttnn/experimental/jit/lazy_utils.hpp"
#include "ttnn/device_operation.hpp"
#include <tt_stl/reflection.hpp>
#include <vector>
#include <memory>
#include <tuple>
#include <utility>
#include <boost/pfr.hpp>

namespace ttnn::experimental::jit {

// Generic wrapper that adapts any device operation to IDeviceOperation
template <typename operation_t>
    requires ttnn::device_operation::DeviceOperationConcept<operation_t>
class LazyDeviceOperation : public LazyOperation {
public:
    using operation_attributes_t = typename operation_t::operation_attributes_t;
    using tensor_args_t = typename operation_t::tensor_args_t;
    using tensor_return_value_t = typename operation_t::tensor_return_value_t;

    LazyDeviceOperation(operation_attributes_t attributes, tensor_args_t tensor_args, const std::string& name) :
        attributes_(std::move(attributes)),
        tensor_args_(tensor_args),  // Copy tensor_args (shallow copy since Tensor is shallow-copyable)
        field_tensor_counts_(extract_field_tensor_counts(tensor_args)),  // Extract tensor counts from parameter
        field_vector_sizes_(extract_field_vector_sizes(tensor_args)),    // Extract vector sizes from parameter
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
        // // Need to make a mutable copy to get non-const iterators for reference binding
        // // TODO: fix from_range_pfr to accept const iterators
        // std::vector<Tensor> mutable_tensors = input_tensors;

        // // Construct tensor_args from input_tensors with proper reference binding
        // // Pass field_tensor_counts_ and field_vector_sizes_ for proper reconstruction
        // auto temp_tensor_args = from_range_pfr<tensor_args_t>(
        //     mutable_tensors.begin(), mutable_tensors.end(), field_tensor_counts_, field_vector_sizes_);

        // // Call the operation's validation method if it exists
        // // Use validate_on_program_cache_miss for comprehensive validation
        // // TODO: Should we choose validate_on_program_cache_miss or validate_on_program_cache_hit based on the cache
        // somehow? if constexpr (requires { operation_t::validate_on_program_cache_miss(attributes_, temp_tensor_args);
        // }) {
        //     operation_t::validate_on_program_cache_miss(attributes_, temp_tensor_args);
        // }
    }

    std::vector<ttnn::TensorSpec> compute_output_specs() const {
        // Return cached output specs that were computed at construction time
        return cached_output_specs_;
    }

    std::string_view name() const override { return std::string_view(name_.c_str()); }

    std::vector<tt::tt_metal::metal_tensor::Tensor> invoke(
        const std::vector<tt::tt_metal::metal_tensor::Tensor>& input_tensors) override {
        // Construct tensor_args from input_tensors with proper reference binding
        // Pass field_tensor_counts_ and field_vector_sizes_ for proper reconstruction
        // TODO: Once Tensor is in metal, we should make all device operations accept metal tensor, not tnn::Tensor
        std::vector<ttnn::Tensor> mutable_tensors;
        mutable_tensors.reserve(input_tensors.size());
        for (const auto& tensor : input_tensors) {
            mutable_tensors.push_back(ttnn::Tensor(tensor));
        }

        // TODO: fix from_range_pfr to accept const iterators
        auto tensor_args = from_range_pfr<tensor_args_t>(
            mutable_tensors.begin(), mutable_tensors.end(), field_tensor_counts_, field_vector_sizes_);

        // Use the standard device operation eager execution path
        auto result = ttnn::device_operation::detail::invoke<operation_t>(attributes_, tensor_args);

        // Convert result to vector
        return convert_result_to_vector(result);
    }

    const operation_attributes_t& get_attributes() const { return attributes_; }

    // TODO: Verify vibecoded to_hash and attributes()
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
    std::vector<size_t> field_tensor_counts_;  // Number of tensors each field consumes
    std::vector<size_t> field_vector_sizes_;   // Total size of vector fields (for vector<optional<T>>)
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

}  // namespace ttnn::experimental::jit
