// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/metal_tensor.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/hal.hpp>

namespace tt::tt_metal {
enum class StorageType;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace ttnn::experimental::lazy {

struct LazyOperation;
struct LazyOperationInputs;

using LazyTensorId = uint32_t;
enum class LazyTensorState {
    LAZY,       // Contains graph node information
    SCHEDULED,  // Operation scheduled for evaluation
    EVALUATED   // Contains actual data
};

class LazyTensor {
    using LazyOperationPtr = std::shared_ptr<ttnn::experimental::lazy::LazyOperation>;
    using LazyOperationInputsPtr = std::shared_ptr<ttnn::experimental::lazy::LazyOperationInputs>;
    using TensorSpec = tt::tt_metal::TensorSpec;
    using MaterializedTensor = tt::tt_metal::metal_tensor::Tensor;

public:
    // Extended tensor spec. Grouped for convenience.
    struct TensorMetadata {
        // Buffer metadata that can be calculated without materializing the buffer
        // TODO: This is error-prone. We have to duplicate changes in logic in an actual buffer to make sure those are
        // aligned. I think we should make metadata calculation logic inside a buffer implementation, and reuse it here.
        struct BufferSpec {
            tt::tt_metal::BufferType buffer_type_ = tt::tt_metal::BufferType::DRAM;
            tt::tt_metal::TensorMemoryLayout buffer_layout_ = tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
            std::optional<tt::tt_metal::ShardSpec> shard_spec_ = std::nullopt;
            tt::tt_metal::DeviceAddr size_ = 0;       // In bytes
            tt::tt_metal::DeviceAddr page_size_ = 0;  // In bytes
            uint32_t element_size_ = 0;
            uint32_t alignment_ = 0;
            tt::tt_metal::DeviceAddr aligned_page_size_ = 0;
            tt::tt_metal::DeviceAddr aligned_size_ = 0;
            bool bottom_up_ = true;  // Allocation direction (default: true for DRAM, false for L1)

            BufferSpec() = default;
            BufferSpec(const tt::tt_metal::TensorSpec& tensor_spec, const tt::tt_metal::MemoryConfig& memory_config);

            static constexpr auto attribute_names = std::forward_as_tuple(
                "buffer_type_",
                "buffer_layout_",
                "shard_spec_",
                "size_",
                "page_size_",
                "element_size_",
                "alignment_",
                "aligned_page_size_",
                "aligned_size_",
                "bottom_up_");
            constexpr auto attribute_values() const {
                return std::forward_as_tuple(
                    this->buffer_type_,
                    this->buffer_layout_,
                    this->shard_spec_,
                    this->size_,
                    this->page_size_,
                    this->element_size_,
                    this->alignment_,
                    this->aligned_page_size_,
                    this->aligned_size_,
                    this->bottom_up_);
            }
        };

        std::optional<TensorSpec> tensor_spec_;  // <- Note really optional, but I need that for default ctor
        tt::tt_metal::distributed::MeshDevice* device_ = nullptr;
        tt::tt_metal::StorageType storage_type_ = tt::tt_metal::StorageType::DEVICE;
        BufferSpec buffer_spec_;

        TensorMetadata() = default;
        TensorMetadata(
            const TensorSpec& tensor_spec,
            tt::tt_metal::distributed::MeshDevice* device,
            tt::tt_metal::StorageType storage_type);
        TensorMetadata(const TensorSpec& tensor_spec, const std::shared_ptr<LazyOperationInputs>& op_inputs);
    };

    LazyTensor() = default;
    LazyTensor(const LazyOperationInputsPtr& op_inputs, const LazyOperationPtr& op, const TensorSpec& tensor_spec);
    LazyTensor(const MaterializedTensor& metal_tensor);

    // This used for ops that return a single tensor
    static std::shared_ptr<LazyTensor> make_lazy_tensor(
        const LazyOperationInputsPtr& op_inputs, const LazyOperationPtr& op, const TensorSpec& tensor_spec);

    // This used for ops that return multiple tensors
    static std::vector<std::shared_ptr<LazyTensor>> make_lazy_tensors(
        const LazyOperationInputsPtr& op_inputs,
        const LazyOperationPtr& op,
        const std::vector<TensorSpec>& tensor_specs);

    static std::shared_ptr<LazyTensor> make_materialized_tensor(const MaterializedTensor& metal_tensor);

    // Inputs and outputs getters
    const LazyOperationInputsPtr& op_inputs() const;
    const std::vector<std::shared_ptr<LazyTensor>>& siblings() const;
    const std::vector<MaterializedTensor>& materialized_tensors() const;
    const MaterializedTensor& materialized_tensor() const;
    MaterializedTensor& materialized_tensor();

    // Tensor metadata getters
    const TensorMetadata::BufferSpec& buffer_spec() const;
    const TensorSpec& tensor_spec() const;
    tt::tt_metal::StorageType storage_type() const;
    tt::tt_metal::distributed::MeshDevice* device() const;

    // Other lazy tensor properties getters
    const LazyOperationPtr& op() const;
    LazyTensorState state() const;
    LazyTensorId id() const;
    bool is_materialized() const;

    void evaluate();

    // Allow optimization passes to modify the graph structure
    // TODO: Maybe we should switch to immutable graph structure? (#31772)
    // TODO: I think we should probably make Pass class friend and make those private
    void set_op_inputs(std::shared_ptr<LazyOperationInputs> new_inputs);
    void set_op(const LazyOperationPtr& new_op);

private:
    void set_siblings(const std::vector<std::shared_ptr<LazyTensor>>& siblings);
    void set_materialized_output_idx(size_t idx);
    void set_state(LazyTensorState state);

    TensorMetadata tensor_metadata_;

    // Links to dependencies and information required to materialize the tensor.
    // TODO: Should we still keep op and inputs in eager mode?
    LazyOperationPtr op_;
    std::shared_ptr<LazyOperationInputs> op_inputs_;
    std::vector<std::shared_ptr<LazyTensor>> siblings_;
    LazyTensorState state_ = LazyTensorState::LAZY;
    LazyTensorId id_ = 0;

    std::vector<MaterializedTensor> materialized_outputs_;
    size_t materialized_output_idx_ = 0;  // In case op produces multiple tensors, we want to know which one is this
};

}  // namespace ttnn::experimental::lazy
