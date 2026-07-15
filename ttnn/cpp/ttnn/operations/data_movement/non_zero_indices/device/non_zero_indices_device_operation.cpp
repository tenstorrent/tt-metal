// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation_types.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_program_factory.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <limits>
#include <tt-metalium/tilize_utils.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {
void NonZeroIndicesDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(
        input_tensor.logical_shape().rank() == 4,
        "non_zero_indices: input must be rank-4, got rank {}",
        input_tensor.logical_shape().rank());
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to Non-zero need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to Non-zero need to be allocated in buffers on device!");

    const uint32_t elem_size = input_tensor.element_size();
    TT_FATAL(
        elem_size == 1 || elem_size == 2 || elem_size == 4,
        "non_zero_indices: unsupported element size {} bytes; only 1, 2, or 4-byte dtypes are supported",
        elem_size);

    TT_FATAL(
        args.output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "non_zero_indices: output_memory_config must be INTERLEAVED; variable-length output cannot be sharded");

    // The ROW_MAJOR kernel uses buffer()->page_size() and num_dev_pages() as loop bounds, which
    // reflect the padded shape.  If padded_shape != logical_shape, padding rows/columns would be
    // scanned and any non-zero padding byte would emit an out-of-range index.  TILE layout handles
    // padding explicitly; for ROW_MAJOR, require no shape padding.
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            input_tensor.padded_shape() == input_tensor.logical_shape(),
            "non_zero_indices: ROW_MAJOR input has shape padding (padded_shape={} vs logical_shape={}); "
            "padded rows/columns are not masked and would produce out-of-range indices",
            input_tensor.padded_shape(),
            input_tensor.logical_shape());
    }

    const uint64_t total_elements = static_cast<uint64_t>(input_tensor.logical_shape().volume());
    TT_FATAL(
        total_elements <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
        "non_zero_indices: input volume {} exceeds uint32_t representable range",
        total_elements);
    // The output indices CB stages all indices in L1 before writing to DRAM.
    // Each non-zero element emits 4 uint32 values (b, n, h, c), so the worst-case
    // CB size is 2 × round_up_to_mul32(volume × 4 × sizeof(uint32_t)).
    const uint64_t aligned_output_bytes =
        static_cast<uint64_t>(round_up_to_mul32(static_cast<uint32_t>(total_elements) * 4 * sizeof(uint32_t)));
    const uint64_t indices_cb_bytes = 2 * aligned_output_bytes;
    const uint64_t l1_size = input_tensor.device()->l1_size_per_core();
    TT_FATAL(
        indices_cb_bytes <= l1_size,
        "non_zero_indices: indices staging buffer requires {} L1 bytes but per-core L1 is {} bytes; "
        "maximum supported volume is ~{} elements",
        indices_cb_bytes,
        l1_size,
        l1_size / (2 * 4 * sizeof(uint32_t)));
}

NonzeroResultSpec NonZeroIndicesDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    ttnn::Shape num_non_zero_shape({1, 1, 1, 8});
    // Output indices tensor is [1, 1, 1, N*4] where each non-zero element stores a (b,n,h,c) 4-tuple
    // of uint32 values, matching torch.nonzero() output format.  Shape must remain a single large
    // page (not [1,1,N,4]) so the NOC write can flush the whole buffer in one {.page_id=0} transfer.
    // Layout is always INTERLEAVED (validated above); honour the caller's requested buffer_type
    // so that memory_config=L1_MEMORY_CONFIG produces an L1 output rather than silently using DRAM.
    const uint32_t flat_n = static_cast<uint32_t>(tensor_args.input.logical_shape().volume());
    const ttnn::Shape indices_shape({1, 1, 1, flat_n * 4});
    const tt::tt_metal::MemoryConfig output_mc{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, args.output_memory_config.buffer_type()};
    TensorLayout layout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), output_mc);
    return {TensorSpec(num_non_zero_shape, layout), TensorSpec(indices_shape, layout)};
}

NonzeroResult NonZeroIndicesDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(std::get<0>(output_specs), tensor_args.input.device()),
        create_device_tensor(std::get<1>(output_specs), tensor_args.input.device()),
    };
}

NonzeroResult nonzero(const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& memory_config) {
    return ttnn::device_operation::launch<NonZeroIndicesDeviceOperation>(
        NonzeroParams{.output_memory_config = memory_config}, NonzeroInputs{.input = input_tensor});
}

}  // namespace ttnn::prim
