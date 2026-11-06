// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <ttnn/distributed/distributed_configs.hpp>
#include <ttnn/types.hpp>
#include <optional>

namespace ttnn {

enum class RandGenerator : uint8_t {
    LFSR = 0,      // hardware PRNG, per-core seeds, position-salted output
    THREEFRY = 1,  // Threefry-2x32 counter-based; output depends only on key and element position
};

Tensor rand(
    const ttnn::Shape& shape,
    MeshDevice& device,
    DataType dtype = DataType::BFLOAT16,
    Layout layout = Layout::TILE,
    const MemoryConfig& memory_config = types::DRAM_MEMORY_CONFIG,
    float from = 0.0f,
    float to = 1.0f,
    uint32_t seed = 0,
    const std::optional<tt::tt_metal::distributed::MeshMapperConfig>& mesh_mapper = std::nullopt,
    RandGenerator generator = RandGenerator::LFSR,
    const std::optional<Tensor>& state = std::nullopt);

// Per-device epoch counters (one uint32 row per core) that rand reads and advances on the device, so a
// captured trace produces fresh values on every replay. Rewrite it to reset the stream.
Tensor rand_state(MeshDevice& device);

}  // namespace ttnn
