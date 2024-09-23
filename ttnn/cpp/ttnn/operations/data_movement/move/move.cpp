// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/move/move.hpp"

#include "device/move_device_operation.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

bool can_deallocate(const Tensor& input_tensor, bool from_multi_device = false) {
    return std::visit(
        [&input_tensor, &from_multi_device](auto&& storage) {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, DeviceStorage>) {
                return storage.buffer.use_count() == (from_multi_device ? 2 : 1);
            } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                bool can_dealloc = true;
                auto input_tensors = get_tensors_from_multi_device_storage(input_tensor);
                for (const auto& device_tensor : input_tensors) {
                    can_dealloc &= can_deallocate(device_tensor, true);
                }
                return can_dealloc;
            } else {
                return false;
            }
        },
        input_tensor.get_storage());
}

static inline Tensor move(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& mem_config) {
    TT_ASSERT(input_tensor.is_allocated(), "Expected input tensor to be allocated");
    auto input_mem_config = input_tensor.memory_config();
    auto input_address = input_tensor.buffer()->address();
    auto output_mem_config = mem_config.value_or(input_mem_config);

    if (not can_deallocate(input_tensor)) {
        // TODO: Should this throw error?
        return input_tensor;
    }

    DeallocateBuffer(*input_tensor.buffer());
    auto output_tensor = create_device_tensor(
        input_tensor.get_legacy_shape(),
        input_tensor.get_dtype(),
        input_tensor.get_layout(),
        input_tensor.device(),
        output_mem_config);

    bool tilized = input_tensor.get_layout() == Layout::TILE;

    // get_parallelization_strategy
    bool move_within_same_mem_space = input_mem_config.buffer_type == output_mem_config.buffer_type;

    // A tensor moved within L1 it is meant to reallocate at higher addresses and a tensor moved within DRAM is meant to
    // reallocate at lower addresses If the tensor is not allocated in a new address, there is no need to move the data
    if (move_within_same_mem_space and input_address == output_tensor.buffer()->address()) {
        tt::log_debug(
            tt::LogOp,
            "WARNING: No space to move the tensor. Move op's input address and output address are equal: {}",
            input_address);
        return output_tensor;
    }

    // Input and output addresses won't overlap if they are in different memory substrates
    bool non_overlap = not move_within_same_mem_space;
    const auto num_banks = input_tensor.device()->num_banks(output_tensor.buffer()->buffer_type());
    uint32_t size_per_bank = tt::tt_metal::detail::SizeBytesPerBank(
        output_tensor.buffer()->size(),
        output_tensor.buffer()->page_size(),
        num_banks,
        output_tensor.buffer()->alignment());

    // If input and output buffers overlap, input has to be copied into circular buffer before writing to output
    // Only compute with storage cores allow CBs to be created
    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    const auto num_l1_banks = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    uint32_t size_per_l1_bank = tt::tt_metal::detail::SizeBytesPerBank(
        output_tensor.buffer()->size(), output_tensor.buffer()->page_size(), num_l1_banks, hal.get_alignment(HalMemType::L1));

    if (move_within_same_mem_space) {
        switch (input_mem_config.buffer_type) {
            // If DRAM, inverse logic because memory is allocated bottom up
            case tt::tt_metal::BufferType::DRAM: {
                non_overlap = output_tensor.buffer()->address() + size_per_bank <= input_address;
            } break;
            case tt::tt_metal::BufferType::L1: {
                non_overlap = input_address + size_per_bank <= output_tensor.buffer()->address();
            } break;
            default: break;
        }
    }

    bool fits_in_cb =
        (output_tensor.device()->get_base_allocator_addr(HalMemType::L1) + size_per_l1_bank) <= (output_mem_config.buffer_type == tt::tt_metal::BufferType::L1
                                                        ? output_tensor.buffer()->address()
                                                        : output_tensor.device()->l1_size_per_core());

    MoveOpParallelizationStrategy move_op_parallelization_strategy = MoveOpParallelizationStrategy::MULTI_CORE;
    if ((not non_overlap) and fits_in_cb and compute_with_storage_grid_size.x > 1 and
        compute_with_storage_grid_size.y > 1) {
        move_op_parallelization_strategy = MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP;
    }

    auto output = operation::run(
                      MoveDeviceOperation{output_mem_config, move_op_parallelization_strategy},
                      {input_tensor, output_tensor},
                      {},
                      {},
                      queue_id)
                      .at(0);
    return output;
}

static inline Tensor move_sharded(
    uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    bool from_multi_device = is_multi_device_tensor(input_tensor);
    operation::launch_op(
        [from_multi_device, mem_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            TT_ASSERT(input_tensor.is_allocated(), "Expected input tensor to be allocated");
            auto input_mem_config = input_tensor.memory_config();
            TT_FATAL(input_mem_config.is_sharded(), "Expected input tensor to be sharded");
            auto input_address = input_tensor.buffer()->address();
            auto output_mem_config = mem_config.value_or(input_mem_config);
            TT_FATAL(output_mem_config.is_sharded(), "Expected output tensor memory config to be sharded");
            if (not can_deallocate(input_tensor, from_multi_device)) {
                TT_FATAL(
                    false,
                    "Expect input tensor to be deallocated after move op. Cannot deallocate before there is probably "
                    "another consumer.");
                // TODO: Should this throw error?
                return {input_tensor};
            }
            auto shard_spec = input_tensor.shard_spec().value();
            auto shard_shape = shard_spec.shape;
            auto shard_grid = shard_spec.grid;
            auto input_shape = input_tensor.get_legacy_shape();
            auto input_dtype = input_tensor.get_dtype();
            auto input_layout = input_tensor.get_layout();

            DeallocateBuffer(*input_tensor.buffer());
            // log_debug(LogOp, "OUTPUT SHARD SPEC: {}", out_shard_spec);
            auto shard_mem_config = output_mem_config;
            shard_mem_config.shard_spec = shard_spec;
            auto output_tensor =
                create_device_tensor(input_shape, input_dtype, input_layout, input_tensor.device(), shard_mem_config);
            if (input_tensor.buffer()->address() == output_tensor.buffer()->address()) {
                tt::log_debug(
                    tt::LogOp,
                    "WARNING: No space to move the tensor. Move op's input address and output address are equal: {}",
                    input_address);
                return {output_tensor};
            }
            MoveOpParallelizationStrategy move_op_parallelization_strategy =
                MoveOpParallelizationStrategy::MULTI_CORE_SHARDED;
            return operation::run(
                MoveDeviceOperation{output_mem_config, move_op_parallelization_strategy},
                {input_tensor, output_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

ttnn::Tensor MoveOperation::invoke(
    uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    if (input_tensor.memory_config().is_sharded()) {
        return move_sharded(queue_id, input_tensor, output_mem_config);
    }
    return move(queue_id, input_tensor, output_mem_config);
}

ttnn::Tensor MoveOperation::invoke(
    const ttnn::Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    return invoke(ttnn::DefaultQueueId, input_tensor, output_mem_config);
}

}  // namespace ttnn::operations::data_movement
