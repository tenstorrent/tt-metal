// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include "tt-metalium/buffer.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::fusion::detail {

// noc_semaphore_inc encodes IND_32 from addr[3:2]; ttsim only implements the
// 16-byte-aligned case (0x107C). Metal CreateSemaphore uses the same stride
// (hal L1 alignment). Each logical bank word owns one 16B slot.
inline constexpr std::uint32_t kFusionSemaphoreStrideBytes = 16;
inline constexpr std::uint32_t kFusionSemaphoreWordsPerSlot =
    kFusionSemaphoreStrideBytes / static_cast<std::uint32_t>(sizeof(std::uint32_t));
static_assert(kFusionSemaphoreStrideBytes % sizeof(std::uint32_t) == 0);

struct FusionSemaphoreBankConfig {
    tt::tt_metal::TensorSpec tensor_spec;
    std::vector<std::uint32_t> initial_values;
};

inline std::optional<FusionSemaphoreBankConfig> make_fusion_semaphore_bank_config(
    const std::vector<tt::tt_metal::CoreRangeSet>& semaphore_core_ranges,
    const std::vector<std::uint32_t>& initial_values) {
    TT_FATAL(
        semaphore_core_ranges.size() == initial_values.size(),
        "Fusion semaphore core-range count ({}) must match initial-value count ({})",
        semaphore_core_ranges.size(),
        initial_values.size());
    if (semaphore_core_ranges.empty()) {
        return std::nullopt;
    }

    tt::tt_metal::CoreRangeSet union_ranges;
    for (const auto& core_ranges : semaphore_core_ranges) {
        TT_FATAL(!core_ranges.empty(), "Fusion semaphore core ranges must not be empty");
        union_ranges = union_ranges.merge(core_ranges);
    }

    const auto num_cores = union_ranges.num_cores();
    const auto num_semaphores = semaphore_core_ranges.size();
    TT_FATAL(num_cores > 0, "Fusion semaphore bank requires at least one participating core");
    TT_FATAL(
        num_cores <= std::numeric_limits<std::uint32_t>::max() &&
            num_semaphores <= std::numeric_limits<std::uint32_t>::max(),
        "Fusion semaphore bank dimensions exceed uint32 limits");
    TT_FATAL(
        num_semaphores <= std::numeric_limits<std::uint32_t>::max() / kFusionSemaphoreWordsPerSlot,
        "Fusion semaphore bank shard width exceeds uint32 limits");
    TT_FATAL(
        num_semaphores <= std::numeric_limits<std::uint32_t>::max() / kFusionSemaphoreStrideBytes,
        "Fusion semaphore bank address offsets exceed uint32 limits");

    const auto num_cores_u32 = static_cast<std::uint32_t>(num_cores);
    const auto shard_width = static_cast<std::uint32_t>(num_semaphores) * kFusionSemaphoreWordsPerSlot;
    const auto shard_spec =
        tt::tt_metal::ShardSpec{union_ranges, {1, shard_width}, tt::tt_metal::ShardOrientation::ROW_MAJOR};
    const auto memory_config = tt::tt_metal::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, shard_spec};

    return FusionSemaphoreBankConfig{
        .tensor_spec = tt::tt_metal::TensorSpec(
            Shape({1, 1, num_cores_u32, shard_width}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                memory_config)),
        .initial_values = initial_values,
    };
}

class FusionSemaphoreBank {
public:
    FusionSemaphoreBank(tt::tt_metal::distributed::MeshDevice* mesh_device, const FusionSemaphoreBankConfig& config) :
        tensor_(allocate_tensor(mesh_device, config.tensor_spec)) {
        TT_FATAL(
            tensor_.device_storage().is_uniform_storage(),
            "Fusion semaphore bank requires storage on every mesh device");
        TT_FATAL(
            !tt::tt_metal::experimental::per_core_allocation::is_per_core_allocation(
                tensor_.mesh_buffer().device_local_config().sharding_args),
            "Fusion semaphore bank requires lockstep allocation; per-core L1 addresses are unsupported");
        const auto base_address = tensor_.mesh_buffer().address();
        for (const auto& device_coord : ttnn::MeshCoordinateRange(mesh_device->shape())) {
            const auto* device_buffer = tensor_.mesh_buffer().get_device_buffer(device_coord);
            TT_FATAL(device_buffer != nullptr, "Fusion semaphore bank is missing storage on a mesh device");
            TT_FATAL(
                device_buffer->address() == base_address,
                "Fusion semaphore bank requires one L1 base address across mesh devices; "
                "found address 0x{:x}, expected 0x{:x}",
                device_buffer->address(),
                base_address);
        }

        const auto num_semaphores = config.initial_values.size();
        TT_FATAL(num_semaphores > 0, "Fusion semaphore bank requires at least one semaphore");
        const auto words_per_core = num_semaphores * kFusionSemaphoreWordsPerSlot;
        const auto volume = config.tensor_spec.logical_shape().volume();
        TT_FATAL(
            volume % words_per_core == 0,
            "Fusion semaphore bank tensor volume ({}) is not divisible by padded semaphore words ({})",
            volume,
            words_per_core);

        std::vector<std::uint32_t> host_values;
        host_values.reserve(volume);
        const auto num_cores = volume / words_per_core;
        for (std::size_t core_index = 0; core_index < num_cores; ++core_index) {
            for (auto initial_value : config.initial_values) {
                host_values.push_back(initial_value);
                host_values.insert(host_values.end(), kFusionSemaphoreWordsPerSlot - 1, 0u);
            }
        }

        Tensor host_tensor(
            tt::tt_metal::HostBuffer(std::move(host_values)),
            config.tensor_spec.logical_shape(),
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::Layout::ROW_MAJOR);
        ttnn::copy_to_device(host_tensor, tensor_);

        TT_FATAL(
            base_address <= std::numeric_limits<std::uint32_t>::max() -
                                static_cast<std::uint32_t>((num_semaphores - 1) * kFusionSemaphoreStrideBytes),
            "Fusion semaphore bank address range overflows uint32");
        addresses_.reserve(num_semaphores);
        for (std::uint32_t index = 0; index < num_semaphores; ++index) {
            addresses_.push_back(base_address + index * kFusionSemaphoreStrideBytes);
        }
    }

    FusionSemaphoreBank(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const std::vector<tt::tt_metal::CoreRangeSet>& semaphore_core_ranges,
        const std::vector<std::uint32_t>& initial_values) :
        FusionSemaphoreBank(
            mesh_device, require_config(make_fusion_semaphore_bank_config(semaphore_core_ranges, initial_values))) {}

    const Tensor& tensor() const { return tensor_; }
    const std::vector<std::uint32_t>& addresses() const { return addresses_; }

private:
    static FusionSemaphoreBankConfig require_config(std::optional<FusionSemaphoreBankConfig> config) {
        TT_FATAL(config.has_value(), "Fusion semaphore bank requires at least one semaphore");
        return std::move(*config);
    }

    static Tensor allocate_tensor(
        tt::tt_metal::distributed::MeshDevice* mesh_device, const tt::tt_metal::TensorSpec& tensor_spec) {
        TT_FATAL(mesh_device != nullptr, "Fusion semaphore bank requires a MeshDevice");
        return ttnn::create_device_tensor(tensor_spec, mesh_device);
    }

    Tensor tensor_;
    std::vector<std::uint32_t> addresses_;
};

}  // namespace ttnn::operations::experimental::fusion::detail
