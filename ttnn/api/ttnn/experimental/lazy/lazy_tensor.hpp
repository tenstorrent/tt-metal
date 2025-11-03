// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/metal_tensor.hpp>
#include <ttnn/tensor/tensor_spec.hpp>

namespace tt::tt_metal {
enum class StorageType;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace ttnn::experimental::lazy {

struct LazyOperation;
using LazyTensorId = uint32_t;
enum class LazyTensorState {
    LAZY,       // Contains graph node information
    SCHEDULED,  // Operation scheduled for evaluation
    EVALUATED   // Contains actual data
};

class LazyTensor {
    using LazyOperationPtr = std::shared_ptr<ttnn::experimental::lazy::LazyOperation>;
    using TensorSpec = tt::tt_metal::TensorSpec;
    using MaterializedTensor = tt::tt_metal::metal_tensor::Tensor;

public:
    LazyTensor() = default;
    LazyTensor(
        const std::vector<std::shared_ptr<LazyTensor>>& op_inputs,
        const LazyOperationPtr& op,
        const TensorSpec& tensor_spec);
    LazyTensor(const MaterializedTensor& metal_tensor);

    // This used for ops that return a single tensor
    static std::shared_ptr<LazyTensor> make_lazy_tensor(
        const std::vector<std::shared_ptr<LazyTensor>>& op_inputs,
        const LazyOperationPtr& op,
        const TensorSpec& tensor_spec);

    // This used for ops that return multiple tensors
    static std::vector<std::shared_ptr<LazyTensor>> make_lazy_tensors(
        const std::vector<std::shared_ptr<LazyTensor>>& op_inputs,
        const LazyOperationPtr& op,
        const std::vector<TensorSpec>& tensor_specs);

    static std::shared_ptr<LazyTensor> make_materialized_tensor(const MaterializedTensor& metal_tensor);

    // Getters
    const std::vector<std::shared_ptr<LazyTensor>>& op_inputs() const;
    const std::vector<std::shared_ptr<LazyTensor>>& siblings() const;
    const std::vector<MaterializedTensor>& materialized_tensors() const;
    const MaterializedTensor& materialized_tensor() const;
    MaterializedTensor& materialized_tensor();
    const TensorSpec& tensor_spec() const;
    LazyTensorState state() const;
    LazyTensorId id() const;
    bool is_materialized() const;
    const LazyOperationPtr& op() const;
    tt::tt_metal::distributed::MeshDevice* device() const;
    tt::tt_metal::StorageType storage_type() const;

    void evaluate();

    // Allow optimization passes to modify the graph structure
    // TODO: Maybe we should switch to immutable graph structure?
    void set_op_inputs(const std::vector<std::shared_ptr<LazyTensor>>& new_inputs);
    void set_op(const LazyOperationPtr& new_op);

private:
    void set_siblings(const std::vector<std::shared_ptr<LazyTensor>>& siblings);
    void set_materialized_output_idx(size_t idx);
    void set_state(LazyTensorState state);

    // Extended tensor spec. Grouped for convenience.
    struct TensorMetadata {
        std::optional<TensorSpec> tensor_spec_;  // <- Note really optional, but I need that for default ctor
        tt::tt_metal::distributed::MeshDevice* device_ = nullptr;
        tt::tt_metal::StorageType storage_type_ = tt::tt_metal::StorageType::DEVICE;

        TensorMetadata() = default;
        TensorMetadata(
            const TensorSpec& tensor_spec,
            tt::tt_metal::distributed::MeshDevice* device,
            tt::tt_metal::StorageType storage_type);
        TensorMetadata(const TensorSpec& tensor_spec, const std::vector<std::shared_ptr<LazyTensor>>& op_inputs);
    };

    TensorMetadata tensor_metadata_;

    // Links to dependencies and information required to materialize the tensor.
    LazyOperationPtr op_;
    std::vector<std::shared_ptr<LazyTensor>> op_inputs_;
    std::vector<std::shared_ptr<LazyTensor>> siblings_;
    LazyTensorState state_ = LazyTensorState::LAZY;
    LazyTensorId id_ = 0;

    std::vector<MaterializedTensor> materialized_outputs_;
    size_t materialized_output_idx_ = 0;  // In case op produces multiple tensors, we want to know which one is this
};

}  // namespace ttnn::experimental::lazy
