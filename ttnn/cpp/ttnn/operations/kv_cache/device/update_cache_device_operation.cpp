// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_cache_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include <algorithm>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

UpdateKVCacheOperation::program_factory_t UpdateKVCacheOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    if (args.op_type == UpdateCacheOpType::FILL) {
        return FillCacheMultiCoreProgramFactory{};
    }
    return UpdateCacheMultiCoreProgramFactory{};
}

void UpdateKVCacheOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE and cache_tensor.storage_type() == StorageType::DEVICE,
        "Operands to update_cache need to be on device!");
    TT_FATAL(input_tensor.device() == cache_tensor.device(), "Operands to update_cache need to be on the same device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr and cache_tensor.buffer() != nullptr,
        "Operands to update_cache need to be allocated in buffers on device!");
    TT_FATAL(
        (input_tensor.layout() == Layout::TILE && cache_tensor.layout() == Layout::TILE),
        "Inputs to update_cache must be tilized");
    TT_FATAL(
        input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
            input_tensor.dtype() == DataType::BFLOAT8_B,
        "Input tensor dtype must be FLOAT32, BFLOAT16, or BFLOAT8_B but got {}",
        input_tensor.dtype());
    TT_FATAL(
        cache_tensor.dtype() == DataType::FLOAT32 || cache_tensor.dtype() == DataType::BFLOAT16 ||
            cache_tensor.dtype() == DataType::BFLOAT8_B,
        "Cache tensor dtype must be FLOAT32, BFLOAT16, or BFLOAT8_B but got {}",
        cache_tensor.dtype());

    TT_FATAL(
        input_tensor.padded_shape()[-1] == cache_tensor.padded_shape()[-1],
        "Input tensor width ({}) must equal cache tensor width ({})",
        input_tensor.padded_shape()[-1],
        cache_tensor.padded_shape()[-1]);
    TT_FATAL(
        input_tensor.padded_shape()[0] == 1,
        "Input tensor batch size must be 1 but got {}",
        input_tensor.padded_shape()[0]);
    TT_FATAL(
        input_tensor.padded_shape()[1] == cache_tensor.padded_shape()[1],
        "Input tensor height ({}) must equal cache tensor height ({})",
        input_tensor.padded_shape()[1],
        cache_tensor.padded_shape()[1]);

    TT_FATAL(
        cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED ||
            (cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
             cache_tensor.memory_config().created_with_nd_shard_spec() &&
             cache_tensor.memory_config()
                 .is_dram()),  // ND_SHARDED layout can collapse to HEIGHT_SHARDED when each bank holds single shard
        "Cache tensor memory layout must be INTERLEAVED, ND_SHARDED, or HEIGHT_SHARDED (created from ND shard spec) "
        "but got {}",
        cache_tensor.memory_config().memory_layout());
    if (args.op_type == UpdateCacheOpType::FILL) {
        // TODO: If we want to support mixed precision like decode, we need to add simple compute kernel for conversion
        TT_FATAL(input_tensor.dtype() == cache_tensor.dtype(), "Input and cache tensors must have same dtype!");

        if (input_tensor.is_sharded()) {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "Input tensor memory layout must not be WIDTH_SHARDED but got {}",
                input_tensor.memory_config().memory_layout());
            TT_FATAL(
                input_tensor.shard_spec().value().shape[1] == input_tensor.padded_shape()[-1],
                "Input tensor shard width ({}) must equal padded width ({})",
                input_tensor.shard_spec().value().shape[1],
                input_tensor.padded_shape()[-1]);
            // Require even work division along seq_len and also only 1 head per core
            TT_FATAL(
                input_tensor.padded_shape()[-2] % input_tensor.shard_spec().value().shape[0] == 0,
                "Seq len must be divisible by shard height!");
        }

        TT_FATAL(
            args.batch_idx < cache_tensor.padded_shape()[0],
            "Batch index ({}) must be less than cache tensor batch size ({})",
            args.batch_idx,
            cache_tensor.padded_shape()[0]);
        TT_FATAL(
            args.update_idx % TILE_HEIGHT == 0,
            "Fill update_idx ({}) must be a multiple of TILE_HEIGHT ({})",
            args.update_idx,
            TILE_HEIGHT);
        TT_FATAL(
            args.update_idx <= cache_tensor.padded_shape()[-2] &&
                input_tensor.padded_shape()[-2] <= cache_tensor.padded_shape()[-2] - args.update_idx,
            "Fill update_idx ({}) + input tensor height ({}) must be <= cache tensor height ({})",
            args.update_idx,
            input_tensor.padded_shape()[-2],
            cache_tensor.padded_shape()[-2]);
    } else if (args.op_type == UpdateCacheOpType::UPDATE) {
        if (input_tensor.is_sharded()) {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "Input tensor memory layout must not be WIDTH_SHARDED but got {}",
                input_tensor.memory_config().memory_layout());
            TT_FATAL(
                input_tensor.shard_spec().value().shape[1] == input_tensor.padded_shape()[-1],
                "Input tensor shard width ({}) must equal padded width ({})",
                input_tensor.shard_spec().value().shape[1],
                input_tensor.padded_shape()[-1]);
            // Require even work division for now
            TT_FATAL(
                (input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) %
                        input_tensor.shard_spec().value().shape[0] ==
                    0,
                "Error");
        } else {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Input tensor memory layout must be INTERLEAVED but got {}",
                input_tensor.memory_config().memory_layout());
        }
        TT_FATAL(
            cache_tensor.padded_shape()[0] <= input_tensor.padded_shape()[-2],
            "Cache tensor batch size ({}) must be <= input tensor height ({})",
            cache_tensor.padded_shape()[0],
            input_tensor.padded_shape()[-2]);
        // batch offset is only valid if num_user less than 32 and batch_offset + num_user <= 32
        if (cache_tensor.padded_shape()[0] < 32) {
            TT_FATAL(
                args.batch_offset + cache_tensor.padded_shape()[0] <= 32,
                "Batch offset ({}) + cache tensor batch size ({}) must be <= 32",
                args.batch_offset,
                cache_tensor.padded_shape()[0]);
        } else {
            TT_FATAL(args.batch_offset == 0, "Batch offset must be 0 when cache tensor batch size >= 32");
        }
    }
}

TensorSpec UpdateKVCacheOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // Do nothing because it's an in-place operation. Cache Tensor is the output tensor.
    return tensor_args.cache.tensor_spec();
}

Tensor UpdateKVCacheOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // Do nothing because it's an in-place operation. Cache Tensor is the output tensor.
    return tensor_args.cache;
}

tt::tt_metal::operation::Hash UpdateKVCacheOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<UpdateKVCacheOperation>(
        args.op_type, std::vector<Tensor>{tensor_args.cache, tensor_args.input});
}

std::vector<tt::tt_metal::DynamicRuntimeArg> UpdateKVCacheOperation::get_dynamic_runtime_args(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-apply, on every cache hit, exactly the runtime args that derive from the hash-excluded
    // attributes (batch_idx / update_idx / batch_offset) OR from a raw baked tensor address. Each
    // branch below MIRRORS the per-core derivation in the corresponding program factory's
    // create_descriptor; if that derivation changes, this MUST change in lockstep.
    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;

    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = cache_tensor.buffer();
    const uint32_t src_addr = src_buffer->address();
    const uint32_t dst_addr = dst_buffer->address();

    tt::tt_metal::IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t Wt = cache_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    if (args.op_type == UpdateCacheOpType::FILL) {
        // --- mirror FillCacheMultiCoreProgramFactory::create_descriptor ---
        // Kernels: reader = 0, writer = 1.
        // Reader arg 0 = src_buffer->address(). Writer arg 0 = dst_buffer->address(),
        // writer arg 2 = cache_start_id (derives from batch_idx + update_idx via start_idx).
        const auto batch_idx = args.batch_idx;
        const auto update_idx = args.update_idx;

        uint32_t num_blocks_of_work =
            input_tensor.padded_shape()[1] * input_tensor.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
        uint32_t input_Ht = input_tensor.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
        uint32_t cache_HtWt = cache_tensor.padded_shape()[-2] * Wt / tt::constants::TILE_HEIGHT;
        uint32_t cache_CHtWt = cache_tensor.padded_shape()[1] * cache_HtWt;
        uint32_t update_idxt = update_idx / tt::constants::TILE_HEIGHT;
        uint32_t start_idx = (batch_idx * cache_CHtWt) + (update_idxt * Wt);

        const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();
        bool row_major;
        uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;
        CoreRangeSet all_cores, core_group_1, core_group_2;
        if (shard_spec.has_value()) {
            row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
            all_cores = shard_spec.value().grid;
            num_cores = all_cores.num_cores();
            core_group_1 = all_cores;
            core_group_2 = CoreRangeSet();
            num_blocks_per_core_group_1 = shard_spec.value().shape[0] / tt::constants::TILE_HEIGHT;
            num_blocks_per_core_group_2 = 0;
            auto bbox = all_cores.bounding_box();
            num_cores_x = bbox.end_coord.x + 1;
            num_cores_y = bbox.end_coord.y + 1;
        } else {
            row_major = true;
            std::tie(
                num_cores,
                all_cores,
                core_group_1,
                core_group_2,
                num_blocks_per_core_group_1,
                num_blocks_per_core_group_2) =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major);
        }

        uint32_t g1_numcores = core_group_1.num_cores();
        const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

        constexpr uint32_t kReaderKernelIdx = 0;
        constexpr uint32_t kWriterKernelIdx = 1;
        dynamic_args.reserve(num_cores * 3);
        for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
            const CoreCoord& core = cores.at(i);
            uint32_t num_blocks_per_core =
                (i < g1_numcores) ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

            // Reader arg 0: src address (raw-baked, not a Buffer* binding) -> re-emit.
            dynamic_args.push_back({kReaderKernelIdx, core, 0, src_addr});

            const uint32_t cache_start_id =
                start_idx + (num_blocks_written / input_Ht * cache_HtWt) + ((num_blocks_written % input_Ht) * Wt);
            // Writer arg 0: dst address (raw-baked) -> re-emit. Writer arg 2: cache_start_id.
            dynamic_args.push_back({kWriterKernelIdx, core, 0, dst_addr});
            dynamic_args.push_back({kWriterKernelIdx, core, 2, cache_start_id});
            num_blocks_written += num_blocks_per_core;
        }
        return dynamic_args;
    }

    // --- mirror UpdateCacheMultiCoreProgramFactory::create_descriptor (UPDATE) ---
    // Kernels: reader = 0, writer = 1, compute kernels (2, [3]) carry NO dynamic args.
    // Reader arg 0 = dst addr, arg 1 = src addr, arg 8 = cache_start_id.
    // Writer arg 0 = dst addr, arg 7 = cache_start_id, arg 10 = tile_update_offset,
    //   arg 11 = batch_read_offset.
    const auto update_idx = args.update_idx;
    const auto batch_offset = args.batch_offset;
    TT_FATAL(args.compute_kernel_config.has_value(), "Compute kernel config is required");
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config.value());

    uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor.padded_shape()[-1] * sizeof(float)
                                       : cache_tensor.padded_shape()[-1] * sizeof(::bfloat16);

    uint32_t cache_total_num_tiles = cache_tensor.physical_volume() / tt::constants::TILE_HW;
    uint32_t cache_batch_num_tiles = cache_total_num_tiles / cache_tensor.padded_shape()[0];
    uint32_t cache_head_num_tiles = cache_batch_num_tiles / cache_tensor.padded_shape()[1];

    uint32_t B = input_tensor.padded_shape()[-2];
    uint32_t num_batched_heads = input_tensor.padded_shape()[1] * B / tt::constants::TILE_HEIGHT;
    uint32_t tile_update_offset = update_idx % tt::constants::TILE_HEIGHT * Wbytes;
    uint32_t batch_read_offset = batch_offset * Wbytes;

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();
    bool row_major;
    uint32_t num_cores, num_batched_heads_per_core_group_1, num_batched_heads_per_core_group_2;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_batched_heads_per_core_group_1 = shard_spec.value().shape[0] / tt::constants::TILE_HEIGHT;
        num_batched_heads_per_core_group_2 = 0;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_batched_heads_per_core_group_1,
            num_batched_heads_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_batched_heads, row_major);
    }

    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    uint32_t cache_tile_idx = update_idx / tt::constants::TILE_HEIGHT * Wt;
    uint32_t total_batched_heads = 0;
    dynamic_args.reserve(num_cores * 7);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_batched_heads_per_core =
            (i < g1_numcores) ? num_batched_heads_per_core_group_1 : num_batched_heads_per_core_group_2;
        uint32_t batch_start_id = (total_batched_heads * tt::constants::TILE_HEIGHT) % B;
        uint32_t cache_start_id = batch_start_id * cache_batch_num_tiles +
                                  ((total_batched_heads * tt::constants::TILE_HEIGHT) / B) * cache_head_num_tiles;
        cache_start_id += cache_tile_idx;

        // Reader: arg 0 = dst addr, arg 1 = src addr (both raw-baked), arg 8 = cache_start_id.
        dynamic_args.push_back({kReaderKernelIdx, core, 0, dst_addr});
        dynamic_args.push_back({kReaderKernelIdx, core, 1, src_addr});
        dynamic_args.push_back({kReaderKernelIdx, core, 8, cache_start_id});
        // Writer: arg 0 = dst addr, arg 7 = cache_start_id, arg 10 = tile_update_offset,
        //   arg 11 = batch_read_offset.
        dynamic_args.push_back({kWriterKernelIdx, core, 0, dst_addr});
        dynamic_args.push_back({kWriterKernelIdx, core, 7, cache_start_id});
        dynamic_args.push_back({kWriterKernelIdx, core, 10, tile_update_offset});
        dynamic_args.push_back({kWriterKernelIdx, core, 11, batch_read_offset});
        total_batched_heads += num_batched_heads_per_core;
    }
    return dynamic_args;
}

Tensor update_cache(
    const Tensor& cache,
    const Tensor& input,
    const uint32_t batch_idx,
    const uint32_t update_index,
    const uint32_t batch_offset,
    const UpdateCacheOpType op_type,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::device_operation::launch<UpdateKVCacheOperation>(
        KvCacheParams{
            .batch_idx = batch_idx,
            .update_idx = update_index,
            .batch_offset = batch_offset,
            .op_type = op_type,
            .compute_kernel_config = compute_kernel_config},
        KvCacheInputs{.cache = cache, .input = input});
}

}  // namespace ttnn::prim
