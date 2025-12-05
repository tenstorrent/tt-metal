// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include "tt_metal/common/core_coord.hpp"

namespace tt::tt_metal::udm {

// Forward declarations
class BlockBuilder;
class BlockProgram;

// ==================================================
//                  Gcore (Global Core)
// ==================================================

/**
 * @brief Global core identifier that abstracts cores across devices
 *
 * A Gcore represents a compute core in the global view, abstracting away
 * the underlying device and local core coordinates.
 */
struct Gcore {
    uint32_t id;  // Global core ID

    bool operator==(const Gcore& other) const { return id == other.id; }
    bool operator!=(const Gcore& other) const { return id != other.id; }
};

// ==================================================
//                  Grid
// ==================================================

/**
 * @brief Virtualized grid abstraction for ND device grids
 *
 * A Grid represents a single device's compute grid in ND dimensions.
 * A grid consists of multiple gcores.
 */
struct Grid {
    std::vector<uint32_t> dimensions;  // ND grid dimensions
    std::vector<Gcore> gcores;         // Global cores in this grid

    Grid() = default;
    explicit Grid(std::vector<uint32_t> dims) : dimensions(std::move(dims)) {}

    size_t num_cores() const { return gcores.size(); }
};

// ==================================================
//                  Block
// ==================================================

/**
 * @brief Virtualized mesh device block
 *
 * A Block represents a virtualized arrangement of mesh devices,
 * allowing users to reshape physical mesh configurations.
 * A block consists of multiple grids (one per device).
 */
struct Block {
    std::vector<uint32_t> dimensions;  // ND block dimensions (e.g., [2, 2, 2] or [1, 8])
    std::vector<Grid> grids;           // Grids in this block (one per device)

    Block() = default;
    explicit Block(std::vector<uint32_t> dims) : dimensions(std::move(dims)) {}

    size_t num_devices() const { return grids.size(); }

    size_t num_blocks() const {
        size_t total = 1;
        for (auto dim : dimensions) {
            total *= dim;
        }
        return total;
    }
};

// ==================================================
//                  GcoresInfo
// ==================================================

/**
 * @brief Information about gcores mapped from a tensor
 *
 * Similar to cores_info in single-device programs, but for global cores.
 */
struct GcoresInfo {
    std::vector<Gcore> gcores;  // List of global cores
    uint32_t num_cores;         // Total number of cores
    uint32_t pages_per_gcore;   // Pages per global core

    // Mapping info for runtime args (no hardcoding needed!)
    std::vector<uint32_t> gcore_to_device_id;         // Maps gcore index → device ID
    std::vector<uint32_t> gcore_to_block_page_start;  // Maps gcore index → starting block page ID
};

// ==================================================
//                  Kernel Handle
// ==================================================

/**
 * @brief Handle for a block kernel
 */
using BlockKernelHandle = uint64_t;

}  // namespace tt::tt_metal::udm
