// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "to_memory_config_op.hpp"

#include "ttnn/core.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/reshard.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_device_operation.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn {

namespace {

bool can_use_sharded_to_interleaved(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<DataType> output_dtype,
    const std::optional<Tensor>& output_tensor) {
    DataType resolved_dtype = output_dtype.value_or(input_tensor.dtype());
    auto [valid, msg] = ttnn::prim::ShardedToInterleavedDeviceOperation::validate_inputs(
        ttnn::prim::ShardedToInterleavedParams{output_mem_config, resolved_dtype},
        ttnn::prim::ShardedToInterleavedInputs{input_tensor, output_tensor});
    if (!valid) {
        return false;
    }

    //***  Check if the CB size is too large for the L1. If so, reroute this to use the default ttnn::copy ***//

    // CB L1 capacity check: output CB is only allocated when dtype conversion is needed
    bool convert_df = input_tensor.dtype() != resolved_dtype;
    if (convert_df) {
        auto shard_spec = input_tensor.shard_spec().value();
        tt::DataFormat output_cb_df = tt::tt_metal::datatype_to_dataformat_converter(resolved_dtype);

        uint32_t output_unit_size = (input_tensor.layout() == Layout::TILE)
                                        ? tt::tile_size(output_cb_df)
                                        : shard_spec.shape[1] * tt::datum_size(output_cb_df);

        uint32_t num_units_per_shard_height = (input_tensor.layout() == Layout::TILE)
                                                  ? shard_spec.shape[0] / input_tensor.tensor_spec().tile().get_height()
                                                  : shard_spec.shape[0];
        uint32_t num_units_per_shard_width = (input_tensor.layout() == Layout::TILE)
                                                 ? shard_spec.shape[1] / input_tensor.tensor_spec().tile().get_width()
                                                 : 1;
        uint32_t num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;

        IDevice* device = input_tensor.device();
        uint32_t output_buffer_alignment = device->allocator()->get_alignment(output_mem_config.buffer_type());
        uint32_t aligned_output_page_size = tt::align(output_unit_size, output_buffer_alignment);

        // Input CB aliases the shard buffer (no extra L1). Output CB is additional.
        uint32_t total_cb_size = num_units_per_shard * aligned_output_page_size;

        uint32_t max_l1_size =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

        if (total_cb_size >= max_l1_size) {
            return false;
        }
    }

    return true;
}

bool can_use_interleaved_to_sharded(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<DataType> output_dtype,
    const std::optional<Tensor>& output_tensor) {
    DataType resolved_dtype = output_dtype.value_or(input_tensor.dtype());
    auto [valid, msg] = ttnn::prim::InterleavedToShardedDeviceOperation::validate_inputs(
        ttnn::prim::InterleavedToShardedParams{output_mem_config, resolved_dtype, false},
        ttnn::prim::InterleavedToShardedInputs{input_tensor, output_tensor});
    if (!valid) {
        return false;
    }

    //***  Check if the CB size is too large for the L1. If so, reroute this to use the default ttnn::copy ***//

    // CB L1 capacity check
    auto shard_spec = output_mem_config.shard_spec().value();
    bool convert_df = input_tensor.dtype() != resolved_dtype;
    bool dst_is_dram = output_mem_config.buffer_type() == BufferType::DRAM;

    IDevice* device = input_tensor.device();
    uint32_t src_alignment = device->allocator()->get_alignment(input_tensor.memory_config().buffer_type());
    uint32_t dst_alignment = device->allocator()->get_alignment(output_mem_config.buffer_type());
    uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    tt::DataFormat input_cb_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_cb_df = tt::tt_metal::datatype_to_dataformat_converter(resolved_dtype);

    uint32_t input_unit_size;
    uint32_t output_unit_size;
    uint32_t num_units_per_shard;
    if (input_tensor.layout() == Layout::TILE) {
        input_unit_size = tt::tile_size(input_cb_df);
        output_unit_size = tt::tile_size(output_cb_df);
        num_units_per_shard = (shard_spec.shape[0] / input_tensor.tensor_spec().tile().get_height()) *
                              (shard_spec.shape[1] / input_tensor.tensor_spec().tile().get_width());
    } else {
        input_unit_size = shard_spec.shape[1] * input_tensor.element_size();
        output_unit_size = shard_spec.shape[1] * tt::datum_size(output_cb_df);
        num_units_per_shard = shard_spec.shape[0];
    }

    // Scratch CB is always created (keep_l1_aligned is hardcoded true in program factory)
    uint32_t scratch_page_size = tt::align(input_unit_size + dram_alignment, dram_alignment);
    constexpr uint32_t num_trids = 4;
    uint32_t total_cb_size = num_trids * scratch_page_size;

    // Input CB: extra L1 only when dtype conversion is needed
    if (convert_df) {
        uint32_t input_page_size = tt::align(input_unit_size, src_alignment);
        total_cb_size += num_units_per_shard * input_page_size;
    }

    // Output CB: aliases shard buffer when dst is L1, extra L1 when dst is DRAM
    if (dst_is_dram) {
        uint32_t output_page_size = tt::align(output_unit_size, dst_alignment);
        total_cb_size += num_units_per_shard * output_page_size;
    }

    uint32_t max_l1_size =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    return total_cb_size < max_l1_size;
}

bool can_use_reshard(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<DataType> output_dtype,
    const std::optional<Tensor>& output_tensor) {
    const auto input_memory_config = ttnn::get_memory_config(input_tensor);
    if (input_memory_config.value().shard_spec().has_value() && output_mem_config.shard_spec().has_value()) {
        const auto input_shard_spec = input_memory_config.value().shard_spec().value();
        const auto output_shard_spec = output_mem_config.shard_spec().value();
        // Check if we can use ttnn::reshard directly
        bool use_reshard_workaround =
            (input_shard_spec.shape[1] != output_shard_spec.shape[1]) &&
            (input_memory_config.value().memory_layout() != output_mem_config.memory_layout() &&
             input_tensor.layout() == Layout::ROW_MAJOR);
        if (use_reshard_workaround) {
            return false;
        }
    }

    if (output_dtype.has_value()) {
        return false;
    }

    auto [valid, msg] = ttnn::prim::ReshardDeviceOperation::validate_inputs(
        ttnn::prim::ReshardParams{output_mem_config}, ttnn::prim::ReshardInputs{input_tensor, output_tensor});
    if (!valid) {
        return false;
    }

    //***  Check if the CB size is too large for the L1. If so, reroute this to use the default ttnn::copy ***//

    // CB L1 capacity checks for reshard program factories
    auto inp_mem_layout = input_tensor.memory_config().memory_layout();
    auto out_mem_layout = output_mem_config.memory_layout();
    auto inp_buffer_type = input_tensor.memory_config().buffer_type();
    auto out_buffer_type = output_mem_config.buffer_type();

    bool has_shard_specs =
        input_tensor.memory_config().shard_spec().has_value() && output_mem_config.shard_spec().has_value();

    bool legacy_reshard = false;
    if (has_shard_specs) {
        if (inp_mem_layout == out_mem_layout && inp_mem_layout != TensorMemoryLayout::BLOCK_SHARDED &&
            inp_mem_layout != TensorMemoryLayout::ND_SHARDED) {
            legacy_reshard = (inp_buffer_type == BufferType::L1 || out_buffer_type == BufferType::L1);
        } else {
            legacy_reshard = (out_buffer_type == BufferType::L1);
        }
    }

    // Same-width reshard (H→H) may allocate a scratch CB when page sizes are unaligned
    if (legacy_reshard && inp_mem_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
        out_mem_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        IDevice* device = input_tensor.device();
        auto inp_shard_spec = input_tensor.memory_config().shard_spec().value();
        auto out_shard_spec = output_mem_config.shard_spec().value();

        bool local_is_output = (out_buffer_type == BufferType::L1);
        auto& remote_shard_spec_ref = local_is_output ? inp_shard_spec : out_shard_spec;
        auto remote_buffer_type = local_is_output ? inp_buffer_type : out_buffer_type;

        tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        uint32_t unit_size;
        uint32_t remote_units_per_shard;
        if (input_tensor.layout() == Layout::TILE) {
            unit_size = tt::tile_size(data_format);
            remote_units_per_shard = remote_shard_spec_ref.numel() / (input_tensor.tensor_spec().tile().get_height() *
                                                                      input_tensor.tensor_spec().tile().get_width());
        } else {
            unit_size = inp_shard_spec.shape[1] * input_tensor.element_size();
            remote_units_per_shard = remote_shard_spec_ref.shape[0];
        }

        uint32_t local_alignment =
            device->allocator()->get_alignment(local_is_output ? out_buffer_type : inp_buffer_type);
        uint32_t remote_alignment = device->allocator()->get_alignment(remote_buffer_type);
        uint32_t local_unit_size_padded = tt::align(unit_size, local_alignment);
        uint32_t remote_unit_size_padded = tt::align(unit_size, remote_alignment);

        bool unaligned = (remote_unit_size_padded != unit_size) || (local_unit_size_padded != unit_size);
        if (unaligned) {
            uint32_t scratch_cb_size = remote_units_per_shard * remote_unit_size_padded;
            uint32_t max_l1_size =
                device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
            if (scratch_cb_size >= max_l1_size) {
                return false;
            }
        }
    }

    // ND reshard copy-pages path (both DRAM) allocates a 1-page CB
    if (!legacy_reshard && inp_buffer_type == BufferType::DRAM && out_buffer_type == BufferType::DRAM) {
        IDevice* device = input_tensor.device();
        uint32_t input_alignment = device->allocator()->get_alignment(BufferType::DRAM);
        tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

        uint32_t page_size;
        if (input_tensor.layout() == Layout::TILE) {
            page_size = tt::tile_size(data_format);
        } else {
            page_size = input_tensor.padded_shape()[-1] * input_tensor.element_size();
        }
        uint32_t aligned_page_size = tt::align(page_size, input_alignment);

        uint32_t max_l1_size =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        if (aligned_page_size >= max_l1_size) {
            return false;
        }
    }

    return true;
}

}  // namespace

Tensor to_memory_config(
    const Tensor& tensor,
    const MemoryConfig& memory_config,
    std::optional<DataType> dtype,
    const std::optional<Tensor>& output_tensor) {
    using namespace tt::tt_metal;

    // Temporary until we see why buffer data not being populated
    const auto original_memory_config = ttnn::get_memory_config(tensor);
    if (original_memory_config.has_value() && original_memory_config.value() == memory_config &&
        !output_tensor.has_value()) {
        return tensor;
    }
    std::vector<std::optional<Tensor>> optional_output_tensors;
    if (output_tensor.has_value()) {
        optional_output_tensors.push_back(output_tensor);
    }

    if (memory_config.is_sharded()) {
        // to_sharded path
        if (tensor.is_sharded()) {
            // reshard
            if (can_use_reshard(tensor, memory_config, dtype, output_tensor)) {
                return ttnn::reshard(tensor, memory_config, output_tensor);
            }
        }
        if (can_use_interleaved_to_sharded(
                tensor,
                memory_config,
                dtype,
                optional_output_tensors.empty() ? std::nullopt : optional_output_tensors.at(0))) {
            const bool keep_l1_aligned = false;
            return ttnn::interleaved_to_sharded(
                tensor,
                memory_config,
                dtype.value_or(tensor.dtype()),
                keep_l1_aligned,
                optional_output_tensors.empty() ? std::nullopt : optional_output_tensors.at(0));
        }
    }
    // to_interleaved path
    if (tensor.is_sharded() && can_use_sharded_to_interleaved(tensor, memory_config, dtype, output_tensor)) {
        return ttnn::prim::sharded_to_interleaved(tensor, memory_config, dtype.value_or(tensor.dtype()), output_tensor);
    }
    // Fallback to ttnn::copy as the default general to_memory_config operation
    return ttnn::prim::copy(
        tensor,
        memory_config,
        dtype.value_or(tensor.dtype()),
        optional_output_tensors.empty() ? std::nullopt : optional_output_tensors.at(0));
}

}  // namespace ttnn
