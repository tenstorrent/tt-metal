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
#include <utility>

namespace ttnn::experimental::lazy {

struct LazyCompositeOperationInputs : public LazyOperationInputs {
    LazyCompositeOperationInputs(const std::vector<Tensor>& input_tensors) : input_tensors_(input_tensors) {}
    void for_each(const std::function<void(const std::shared_ptr<LazyTensor>&)>& fn) const override {
        for (const auto& tensor : input_tensors_) {
            fn(tensor.lazy());
        }
    }
    std::any inputs() const override { return input_tensors_; }
private:
    std::vector<Tensor> input_tensors_;
};

// Generic wrapper that adapts any device operation to LazyOperation
template <typename operation_t>
class LazyCompositeOperation : public LazyOperation {
public:
    using tensor_return_value_t = typename operation_t::tensor_return_value_t;

    LazyCompositeOperation(operation_t operation, const std::string& name) : name_(name), operation_(operation) {}

    std::string_view name() const override { return std::string_view(name_.c_str()); }

    std::vector<tt::tt_metal::metal_tensor::Tensor> invoke(
        const LazyOperationInputs& inputs) override {
        // Use the standard device operation eager execution path
        // TODO: Remove this conversion once we have a proper metal tensor
        auto input_tensors = std::any_cast<std::vector<Tensor>>(inputs.inputs());
        return convert_result_to_vector(operation_.invoke(input_tensors));
    }

    tt::stl::hash::hash_t operation_type_id() const override { return tt::stl::hash::type_hash<operation_t>; }

    operation_t& operation() { return operation_; }

private:
    std::string name_;
    operation_t operation_;

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
std::shared_ptr<LazyCompositeOperation<operation_t>> make_lazy_composite_operation(
    operation_t operation, const std::string& name) {
    return std::make_shared<LazyCompositeOperation<operation_t>>(operation, name);
}

template <typename operation_t>
std::shared_ptr<LazyCompositeOperationInputs> make_lazy_composite_operation_inputs(
    const std::vector<Tensor>& input_tensors) {
    return std::make_shared<LazyCompositeOperationInputs>(input_tensors);
}

}  // namespace ttnn::experimental::lazy
