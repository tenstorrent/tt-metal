// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dummy_op_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dummy_op {

namespace {

bool is_dram_interleaved(const ttnn::Tensor& tensor) {
    const auto& mem_cfg = tensor.memory_config();
    return mem_cfg.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem_cfg.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

}  // namespace

void DummyOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "input_tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "input_tensor must have a buffer");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE,
        "input_tensor must be TILE layout, got {}",
        input_tensor.layout());
    TT_FATAL(is_dram_interleaved(input_tensor), "input_tensor must be DRAM interleaved");
    TT_FATAL(operation_attributes.num_iter > 0, "num_iter must be > 0");

    // Kernels assume a single row of cores (one core per x-index, all with the
    // same y). Anything else would break the per-core split formula and the
    // reader/writer's address arithmetic.
    const auto& crs = operation_attributes.worker_core_range_set;
    TT_FATAL(crs.size() == 1, "worker_core_range_set must contain exactly one CoreRange, got {}", crs.size());
    const auto& range = *crs.ranges().begin();
    TT_FATAL(
        range.start_coord.y == range.end_coord.y,
        "worker_core_range_set must span exactly one Tensix row (start.y == end.y); got start.y={}, end.y={}",
        range.start_coord.y,
        range.end_coord.y);

    // Each core's tile count must be a multiple of the kernel's batch size so
    // pushes tile the CB cleanly.
    constexpr uint32_t kBatchSize = 8;
    const uint32_t num_tiles = input_tensor.buffer()->num_pages();
    const uint32_t num_cores = crs.num_cores();
    TT_FATAL(
        num_tiles % (num_cores * kBatchSize) == 0,
        "input_tensor tile count ({}) must be divisible by num_cores*batch ({}*{} = {}); "
        "otherwise some core's per-iter push pattern includes a tail of <{} tiles, which "
        "leaves the CB's fifo_wr_ptr at a non-aligned offset and lets the next full-batch "
        "push overshoot fifo_limit (cb_push_back assert at dataflow_api.h:218).",
        num_tiles,
        num_cores,
        kBatchSize,
        num_cores * kBatchSize,
        kBatchSize);
}

void DummyOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

DummyOpDeviceOperation::spec_return_value_t DummyOpDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    return tensor_args.input_tensor.tensor_spec();
}

// In-place op: return the input tensor itself so the writer writes back to the
// same DRAM addresses it read from.
DummyOpDeviceOperation::tensor_return_value_t DummyOpDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    return tensor_args.input_tensor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dummy_op

namespace ttnn::prim {

ttnn::Tensor prefill_dummy_op(
    const ttnn::Tensor& input_tensor,
    uint32_t num_iter,
    const CoreRangeSet& worker_core_range_set,
    uint32_t global_semaphore_address) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::dummy_op::DummyOpDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .num_iter = num_iter,
            .worker_core_range_set = worker_core_range_set,
            .global_semaphore_address = global_semaphore_address},
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}

}  // namespace ttnn::prim
