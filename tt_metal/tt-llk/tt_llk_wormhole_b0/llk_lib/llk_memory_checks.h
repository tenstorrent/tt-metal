// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "core_config.h"
#include "dev_mem_map.h"

/**
 * @brief Transform L1 address to the format used by LLK functions
 *
 * This transformation consists of two parts:
 *
 * 1. Division by 16 (shift right by 4 bits):
 *    Hardware L1 address fields use 16-byte granularity, not byte addresses.
 *    The hardware expects addresses in units of 16-byte blocks, so we convert
 *    byte addresses to 16-byte-aligned addresses by dividing by 16.
 *
 * 2. Decrement by 1 (on Tensix architectures like Wormhole/Blackhole):
 *    Tensix L1 address fields use an off-by-one convention: you program (addr_16B - 1),
 *    and the hardware internally increments the value before using it. This hardware
 *    quirk requires the subtraction of 1 in the address transformation.
 *
 * @param buffer_address The physical L1 address (byte-aligned)
 * @return Transformed address for LLK use: (address / 16) - 1
 */
constexpr inline std::uint32_t L1_ADDRESS(std::uint32_t buffer_address)
{
    return (buffer_address >> 4) - 1;
}

namespace ckernel
{

// L1 Memory Layout for Tile Data Verification
// ====================================================================
//
// This file defines the valid memory regions in L1 where tile data can be safely stored.
// The layout is based on dev_mem_map.h hardware definitions and excludes all reserved
// system memory areas.
//
// WORMHOLE: 4-directional fabric
//
// Memory Region Breakdown (from dev_mem_map.h):
// -----------------------------------------------
// MEM_L1_BASE - MEM_MAP_END: RESERVED - System firmware, mailboxes, counters, routing
//                                       tables, fabric metadata, and packet header pools
//
// MEM_MAP_END - (MEM_L1_BASE + MEM_L1_SIZE): AVAILABLE - Valid memory for tile data buffers
//                                                        This is the primary usable L1 region for computation
//
// MEM_MAP_END calculation chain (all from dev_mem_map.h):
// ---------------------------------------------------------
//   MEM_MAILBOX_BASE + MEM_MAILBOX_SIZE = MEM_MAILBOX_END
//   → MEM_ZEROS_BASE = ((MEM_MAILBOX_END + alignment) & ~alignment)
//   → MEM_BRISC_FIRMWARE_BASE = MEM_ZEROS_BASE + MEM_ZEROS_SIZE
//   → MEM_NCRISC_FIRMWARE_BASE = MEM_BRISC_FIRMWARE_BASE + MEM_BRISC_FIRMWARE_SIZE
//   → MEM_TRISC0_FIRMWARE_BASE = MEM_NCRISC_FIRMWARE_BASE + MEM_NCRISC_FIRMWARE_SIZE
//   → MEM_TRISC1_FIRMWARE_BASE = MEM_TRISC0_FIRMWARE_BASE + MEM_TRISC0_FIRMWARE_SIZE
//   → MEM_TRISC2_FIRMWARE_BASE = MEM_TRISC1_FIRMWARE_BASE + MEM_TRISC1_FIRMWARE_SIZE
//   → MEM_MAP_END = MEM_TRISC2_FIRMWARE_BASE + MEM_TRISC2_FIRMWARE_SIZE

// Start of available memory region for tile data (LLK transformed address)
// Immediately after all reserved system memory (firmware, mailboxes, counters, routing tables, fabric metadata)
// Physical Boundary: MEM_MAP_END
// Transformed Value: L1_ADDRESS(MEM_MAP_END) = (MEM_MAP_END / 16) - 1
constexpr std::uint32_t L1_REGION_START = L1_ADDRESS(MEM_MAP_END);

// End of L1 memory (LLK transformed address) - total available L1 size
// This is the absolute upper bound for any L1 address
// Physical Boundary: MEM_L1_BASE + MEM_L1_SIZE
// Transformed Value: L1_ADDRESS(MEM_L1_BASE + MEM_L1_SIZE) = ((MEM_L1_BASE + MEM_L1_SIZE) / 16) - 1
constexpr std::uint32_t L1_REGION_END = L1_ADDRESS(MEM_L1_BASE + MEM_L1_SIZE);

} // namespace ckernel

/**
 * @brief Check if an LLK-transformed address is valid for tile data
 *
 * Validates that the transformed address falls within the usable L1 memory region:
 * - Start (transformed): L1_ADDRESS(MEM_MAP_END)
 * - End (transformed):   L1_ADDRESS(MEM_L1_BASE + MEM_L1_SIZE)
 * - Physical Range: MEM_MAP_END to (MEM_L1_BASE + MEM_L1_SIZE)
 *
 * This single contiguous region is available for all tile data and computational buffers.
 * The reserved area (MEM_L1_BASE to MEM_MAP_END) contains system firmware, mailboxes, counters,
 * routing tables, fabric metadata, and packet header pools.
 *
 * IMPORTANT: This function takes and compares LLK TRANSFORMED addresses.
 * If you have a physical address, transform it first using L1_ADDRESS():
 *   transformed_addr = L1_ADDRESS(physical_addr) = (physical_addr / 16) - 1
 * Or pass this function the result from L1_ADDRESS() calls.
 *
 * @param address The LLK-transformed L1 address to validate
 * @return true if address is within valid tile data region [L1_REGION_START, L1_REGION_END)
 */
inline static bool is_valid_L1_address(const std::uint32_t address)
{
    return (address >= ckernel::L1_REGION_START && address < ckernel::L1_REGION_END);
}
