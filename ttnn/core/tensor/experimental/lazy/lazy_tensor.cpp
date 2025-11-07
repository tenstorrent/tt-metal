// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/tensor/tensor.hpp>
#include "ttnn/experimental/lazy/lazy_tensor.hpp"
#include "ttnn/experimental/lazy/graph_utils.hpp"
#include "ttnn/experimental/lazy/lazy_operation.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::experimental::lazy {

namespace {
// Calculate alignment from buffer type using HAL
uint32_t calculate_alignment(tt::tt_metal::BufferType buffer_type) {
    switch (buffer_type) {
        case tt::tt_metal::BufferType::DRAM:
        case tt::tt_metal::BufferType::TRACE: return tt::tt_metal::hal::get_dram_alignment();
        case tt::tt_metal::BufferType::L1:
        case tt::tt_metal::BufferType::L1_SMALL: return tt::tt_metal::hal::get_l1_alignment();
        case tt::tt_metal::BufferType::SYSTEM_MEMORY: return tt::tt_metal::hal::get_pcie_alignment();
        default: TT_THROW("Unsupported buffer type!");
    }
}
}  // anonymous namespace

// Lazy Tensor
LazyTensor::LazyTensor(
    const LazyOperationInputsPtr& op_inputs, const LazyOperationPtr& op, const TensorSpec& tensor_spec) :
    op_inputs_(op_inputs),
    op_(op),
    tensor_metadata_(tensor_spec, op_inputs),
    id_(GraphUtils::get_available_lazy_tensor_id()) {}

std::shared_ptr<LazyTensor> LazyTensor::make_lazy_tensor(
    const LazyOperationInputsPtr& op_inputs, const LazyOperationPtr& op, const TensorSpec& tensor_spec) {
    return std::make_shared<LazyTensor>(op_inputs, op, tensor_spec);
}

std::shared_ptr<LazyTensor> LazyTensor::make_materialized_tensor(
    const tt::tt_metal::metal_tensor::Tensor& metal_tensor) {
    return std::make_shared<LazyTensor>(metal_tensor);
}

std::vector<std::shared_ptr<LazyTensor>> LazyTensor::make_lazy_tensors(
    const LazyOperationInputsPtr& op_inputs, const LazyOperationPtr& op, const std::vector<TensorSpec>& tensor_specs) {
    std::vector<std::shared_ptr<LazyTensor>> lazy_tensors;
    lazy_tensors.reserve(tensor_specs.size());
    for (const auto& tensor_spec : tensor_specs) {
        lazy_tensors.push_back(make_lazy_tensor(op_inputs, op, tensor_spec));
    }
    for (size_t i = 0; i < lazy_tensors.size(); i++) {
        std::vector<std::shared_ptr<LazyTensor>> siblings;
        siblings.reserve(lazy_tensors.size() - 1);
        for (size_t j = 0; j < lazy_tensors.size(); j++) {
            if (i != j) {
                siblings.push_back(lazy_tensors[j]);
            }
        }
        lazy_tensors[i]->set_siblings(siblings);
        lazy_tensors[i]->set_materialized_output_idx(i);
    }
    return lazy_tensors;
}

LazyTensor::LazyTensor(const tt::tt_metal::metal_tensor::Tensor& metal_tensor) :
    op_inputs_({}),
    op_(nullptr),
    tensor_metadata_(metal_tensor.tensor_spec(), metal_tensor.device(), metal_tensor.storage_type()),
    siblings_({}),
    materialized_outputs_({metal_tensor}),
    materialized_output_idx_(0),
    state_(LazyTensorState::EVALUATED),
    id_(GraphUtils::get_available_lazy_tensor_id()) {}

// Getters
const LazyTensor::LazyOperationInputsPtr& LazyTensor::op_inputs() const { return op_inputs_; }
const std::vector<std::shared_ptr<LazyTensor>>& LazyTensor::siblings() const { return siblings_; }
const std::vector<tt::tt_metal::metal_tensor::Tensor>& LazyTensor::materialized_tensors() const {
    return materialized_outputs_;
}
const tt::tt_metal::metal_tensor::Tensor& LazyTensor::materialized_tensor() const {
    return materialized_outputs_[materialized_output_idx_];
}
tt::tt_metal::metal_tensor::Tensor& LazyTensor::materialized_tensor() {
    return materialized_outputs_[materialized_output_idx_];
}

const TensorSpec& LazyTensor::tensor_spec() const { return tensor_metadata_.tensor_spec_.value(); }
tt::tt_metal::StorageType LazyTensor::storage_type() const { return tensor_metadata_.storage_type_; }
tt::tt_metal::distributed::MeshDevice* LazyTensor::device() const { return tensor_metadata_.device_; }
uint32_t LazyTensor::buffer_alignment() const { return tensor_metadata_.buffer_metadata_.alignment_; }
tt::tt_metal::DeviceAddr LazyTensor::buffer_page_size() const { return tensor_metadata_.buffer_metadata_.page_size_; }
tt::tt_metal::DeviceAddr LazyTensor::buffer_aligned_page_size() const {
    return tensor_metadata_.buffer_metadata_.aligned_page_size_;
}
tt::tt_metal::DeviceAddr LazyTensor::buffer_size() const { return tensor_metadata_.buffer_metadata_.size_; }
tt::tt_metal::DeviceAddr LazyTensor::buffer_aligned_size() const {
    return tensor_metadata_.buffer_metadata_.aligned_size_;
}
tt::tt_metal::BufferType LazyTensor::buffer_type() const { return tensor_metadata_.buffer_metadata_.buffer_type_; }
tt::tt_metal::TensorMemoryLayout LazyTensor::buffer_layout() const {
    return tensor_metadata_.buffer_metadata_.buffer_layout_;
}
bool LazyTensor::buffer_bottom_up() const { return tensor_metadata_.buffer_metadata_.bottom_up_; }

const LazyTensor::LazyOperationPtr& LazyTensor::op() const { return op_; }
LazyTensorState LazyTensor::state() const { return state_; }
LazyTensorId LazyTensor::id() const { return id_; }
bool LazyTensor::is_materialized() const { return state_ == LazyTensorState::EVALUATED; }

void LazyTensor::evaluate() {
    if (state_ == LazyTensorState::EVALUATED || state_ == LazyTensorState::SCHEDULED) {
        return;
    }

    state_ = LazyTensorState::SCHEDULED;

    // Verify that all inputs are materialized
    op_inputs_->for_each([&](const std::shared_ptr<LazyTensor>& input) {
        auto op_name = input->op() ? input->op()->name() : "Unknown";
        TT_FATAL(
            input->is_materialized(),
            "Input tensor {} produced by operation {} is not materialized",
            input->id(),
            op_name);
    });

    materialized_outputs_ = op_->invoke(*op_inputs_);
    state_ = LazyTensorState::EVALUATED;
    // Now update siblings' materialized tensors
    for (auto& sibling : siblings_) {
        // TODO: make sure that this works
        sibling->materialized_outputs_ = materialized_outputs_;
        sibling->state_ = LazyTensorState::EVALUATED;
    }
}

void LazyTensor::set_siblings(const std::vector<std::shared_ptr<LazyTensor>>& siblings) { siblings_ = siblings; }
void LazyTensor::set_materialized_output_idx(size_t idx) { materialized_output_idx_ = idx; }
void LazyTensor::set_state(LazyTensorState state) { state_ = state; }

void LazyTensor::set_op_inputs(std::shared_ptr<LazyOperationInputs> new_inputs) { op_inputs_ = std::move(new_inputs); }
void LazyTensor::set_op(const LazyOperationPtr& new_op) { op_ = new_op; }

// ======================= LazyTensor::TensorMetadata =======================
LazyTensor::TensorMetadata::TensorMetadata(
    const TensorSpec& tensor_spec,
    tt::tt_metal::distributed::MeshDevice* device,
    tt::tt_metal::StorageType storage_type) :
    tensor_spec_(tensor_spec), device_(device), storage_type_(storage_type) {
    buffer_metadata_ = Buffer(tensor_spec, tensor_spec.memory_config());
}

LazyTensor::TensorMetadata::TensorMetadata(
    const TensorSpec& tensor_spec, const std::shared_ptr<LazyOperationInputs>& op_inputs) :
    tensor_spec_(tensor_spec) {
    // Inherit device and storage_type from first input tensor
    // TODO: Is this correct though?
    size_t index = 0;
    op_inputs->for_each([&](const std::shared_ptr<LazyTensor>& input) {
        // TODO: This is a bit inefficient. maybe we should just pass device and storage type in lazy tesor ctor
        if (index == 0) {
            device_ = input->device();
            storage_type_ = input->storage_type();
        }
        index++;
    });
    if (index == 0) {
        device_ = nullptr;
        storage_type_ = tt::tt_metal::StorageType::DEVICE;
    }

    buffer_metadata_ = Buffer(tensor_spec, tensor_spec.memory_config());
}

// ======================= LazyTensor::TensorMetadata::BufferMetadata =======================
LazyTensor::TensorMetadata::Buffer::Buffer(
    const tt::tt_metal::TensorSpec& tensor_spec, const tt::tt_metal::MemoryConfig& memory_config) :
    buffer_type_(memory_config.buffer_type()),
    buffer_layout_(memory_config.memory_layout()),
    shard_spec_(memory_config.shard_spec()),
    element_size_(tt::tt_metal::tensor_impl::element_size_bytes(tensor_spec.data_type())),
    size_(tensor_spec.compute_packed_buffer_size_bytes()),
    page_size_(tensor_spec.compute_page_size_bytes()),
    alignment_(calculate_alignment(memory_config.buffer_type())),
    aligned_page_size_(tt::round_up(page_size_, static_cast<tt::tt_metal::DeviceAddr>(alignment_))),
    bottom_up_(
        (memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM ||
         memory_config.buffer_type() == tt::tt_metal::BufferType::TRACE)) {
    // Calculate aligned size
    // For aligned size, we need number of pages
    if (page_size_ > 0) {
        uint32_t num_pages = (size_ + page_size_ - 1) / page_size_;  // ceiling division
        aligned_size_ = num_pages * aligned_page_size_;
    } else {
        aligned_size_ = 0;
    }
}

}  // namespace ttnn::experimental::lazy
