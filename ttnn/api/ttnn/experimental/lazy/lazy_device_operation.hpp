// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/lazy/lazy_operation.hpp"
#include "ttnn/experimental/lazy/lazy_operation_inputs.hpp"
#include "ttnn/experimental/lazy/lazy_utils.hpp"
#include "ttnn/device_operation.hpp"
#include <tt_stl/reflection.hpp>
#include <vector>
#include <memory>

namespace ttnn::experimental::lazy {

template <typename operation_t>
class LazyDeviceOperationInputs : public LazyOperationInputs {
public:
    using tensor_args_t = typename operation_t::tensor_args_t;
    LazyDeviceOperationInputs(const tensor_args_t& tensor_args) : tensor_args_(tensor_args) {}
    void for_each(const std::function<void(const std::shared_ptr<LazyTensor>&)>& fn) const override {
        tt::stl::reflection::visit_object_of_type<Tensor>([&](const Tensor& t) { fn(t.lazy()); }, tensor_args_);
    }
    std::any get() const override { return tensor_args_; }

private:
    tensor_args_t tensor_args_;
};

// Generic wrapper that adapts any device operation to LazyOperation
template <typename operation_t>
    requires ttnn::device_operation::DeviceOperationConcept<operation_t>
class LazyDeviceOperation : public LazyOperation {
public:
    using operation_attributes_t = typename operation_t::operation_attributes_t;
    using tensor_args_t = typename operation_t::tensor_args_t;
    using tensor_return_value_t = typename operation_t::tensor_return_value_t;

    LazyDeviceOperation(operation_attributes_t attributes, const tensor_args_t& tensor_args, const std::string& name) :
        attributes_(attributes), name_(name) {}

    std::string_view name() const override { return std::string_view(name_.c_str()); }

    std::vector<tt::tt_metal::metal_tensor::Tensor> invoke(const LazyOperationInputs& inputs) override {
        // Reconstruct tensor_args from input tensors using stored metadata
        auto tensor_args = std::any_cast<tensor_args_t>(inputs.get());

        // Use the standard device operation eager execution path
        auto result = ttnn::device_operation::detail::invoke<operation_t>(attributes_, tensor_args);

        // Convert result to vector
        return convert_result_to_vector(result);
    }

    const operation_attributes_t& attributes() const { return attributes_; }

    tt::stl::hash::hash_t operation_type_id() const override { return tt::stl::hash::type_hash<operation_t>; }

    std::vector<std::optional<TensorSpec>> compute_output_specs(const tensor_args_t& tensor_args) const {
        auto specs = operation_t::compute_output_specs(attributes_, tensor_args);
        return flatten_specs(specs);
    }

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
    std::string name_;

    // Helper to extract metal tensor from Tensor or optional<Tensor>
    static std::optional<tt::tt_metal::metal_tensor::Tensor> extract_metal_tensor(const auto& value) {
        using value_t = std::decay_t<decltype(value)>;
        if constexpr (std::same_as<value_t, Tensor>) {
            return value.get_materialized_tensor();
        } else if constexpr (std::same_as<value_t, std::optional<Tensor>>) {
            if (value.has_value()) {
                return value->get_materialized_tensor();
            }
            return std::nullopt;
        } else {
            static_assert(std::same_as<value_t, void>, "Unsupported tensor element type");
            return std::nullopt;
        }
    }

    // Convert result from device operation to vector of tensors
    std::vector<tt::tt_metal::metal_tensor::Tensor> convert_result_to_vector(
        const tensor_return_value_t& result) const {
        // TODO: device operations are supposed to return metal tensors, not ttnn::Tensor

        // Single element (Tensor or optional<Tensor>)
        if constexpr (
            std::same_as<tensor_return_value_t, Tensor> || std::same_as<tensor_return_value_t, std::optional<Tensor>>) {
            auto metal_tensor = extract_metal_tensor(result);
            if (metal_tensor.has_value()) {
                return {metal_tensor.value()};
            }
            return {};
        }
        // Tuple-like (array or tuple) - convert to vector
        else if constexpr (TupleLike<tensor_return_value_t>) {
            constexpr auto N = std::tuple_size_v<tensor_return_value_t>;
            std::vector<tt::tt_metal::metal_tensor::Tensor> metal_tensors;

            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                (
                    [&] {
                        auto metal_tensor = extract_metal_tensor(std::get<Is>(result));
                        if (metal_tensor.has_value()) {
                            metal_tensors.push_back(metal_tensor.value());
                        }
                    }(),
                    ...);
            }(std::make_index_sequence<N>{});

            return metal_tensors;
        }
        // Range-like (vector, etc) - iterate and convert
        else if constexpr (RangeLike<tensor_return_value_t>) {
            std::vector<tt::tt_metal::metal_tensor::Tensor> metal_tensors;
            for (const auto& elem : result) {
                auto metal_tensor = extract_metal_tensor(elem);
                if (metal_tensor.has_value()) {
                    metal_tensors.push_back(metal_tensor.value());
                }
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

template <typename operation_t>
std::shared_ptr<LazyOperationInputs> make_lazy_device_operation_inputs(
    const typename operation_t::tensor_args_t& tensor_args) {
    return std::make_shared<LazyDeviceOperationInputs<operation_t>>(tensor_args);
}
}  // namespace ttnn::experimental::lazy
