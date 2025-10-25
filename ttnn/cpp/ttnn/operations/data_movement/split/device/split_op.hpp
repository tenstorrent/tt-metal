// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/experimental/jit/IDeviceOperation.hpp"
#include "tt_stl/reflection.hpp"

namespace ttnn::operations::data_movement {

struct SplitDeviceOperation : public ttnn::experimental::jit::IDeviceOperation {
    const int num_splits;
    const int dim;
    const tt::tt_metal::MemoryConfig output_mem_config;

    // Constructor
    SplitDeviceOperation(const int num_splits, const int dim, const tt::tt_metal::MemoryConfig& output_mem_config) :
        num_splits(num_splits), dim(dim), output_mem_config(output_mem_config) {}

    // Required for reflection/hashing
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("num_splits", num_splits);
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("output_mem_config", output_mem_config);

        return attrs;
    }

    tt::stl::hash::hash_t to_hash() const {
        return tt::stl::hash::hash_objects_with_default_seed(num_splits, dim, output_mem_config);
    }

    std::vector<Tensor> invoke(std::vector<Tensor> input_tensors) override;

    void validate(const std::vector<ttnn::experimental::jit::LazyTensor>& input_tensors) const override;
    std::vector<ttnn::TensorSpec> compute_output_specs(
        const std::vector<ttnn::experimental::jit::LazyTensor>& input_tensors) const override;

    void validate(const std::vector<Tensor>& input_tensors) const override;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const override;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const override;
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const override;
    std::vector<Tensor> get_output_tensors() const override;
    void set_output_tensors(std::vector<Tensor> output_tensors) override;

private:
    std::vector<Tensor> output_tensors_;
};

}  // namespace ttnn::operations::data_movement
