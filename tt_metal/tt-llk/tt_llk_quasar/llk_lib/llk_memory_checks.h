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
 * Hardware L1 address fields use 16-byte granularity, not byte addresses.
 * The hardware expects addresses in units of 16-byte blocks, so we convert
 * byte addresses to 16-byte-aligned addresses by dividing by 16.
 *
 * For quasar, tensix L1 address fields do not use an off-by-one convention,
 * therefore, no -1 is needed in the transformation (like in Wormhole/Blackhole).
 *
 * @param buffer_address The physical L1 address (byte-aligned)
 * @return Transformed address for LLK use: address / 16
 */
constexpr inline std::uint32_t L1_ADDRESS(std::uint32_t buffer_address)
{
    return buffer_address >> 4;
}

namespace ckernel
{

// L1 Memory Layout for Tile Data Verification (Quasar Architecture)
// ==================================================================
//
// This file defines the valid memory regions in L1 where tile data can be safely stored.
// The layout is based on dev_mem_map.h hardware definitions and excludes all reserved
// system memory areas.
//
// QUASAR: Significantly larger L1 memory (4 MB) with 8 DM cores and 4 TRISC cores
// Much more reserved memory for firmware, DM kernels, global/local storage compared to previous generations
//
// Memory Region Breakdown (from dev_mem_map.h):
// -----------------------------------------------
// MEM_L1_BASE - MEM_MAP_END: RESERVED - System firmware (mailbox, debug, DM/TRISC FW),
//                                       global storage (8 DM cores + 4 TRISC cores),
//                                       local storage (8 DM cores), DM kernel code,
//                                       NoC/Fabric counters, routing tables, and packet header pools
//                                       Calculated as: MEM_MAP_END
//
// MEM_MAP_END - (MEM_L1_BASE + MEM_L1_SIZE): AVAILABLE - Valid memory for tile data buffers
//                                                This is the primary usable L1 region for computation
//
// Total L1: MEM_L1_SIZE
//
// Exact Value Calculations (all from dev_mem_map.h chain):
// ---------------------------------------------------------
// MEM_MAP_END calculation chain (Quasar specific with 8 DM cores):
//   MEM_MAILBOX_BASE + MEM_MAILBOX_SIZE = MEM_MAILBOX_END
//   → MEM_ZEROS_BASE = ((MEM_MAILBOX_END + alignment) & ~alignment)
//   → MEM_LLK_DEBUG_BASE = MEM_ZEROS_BASE + MEM_ZEROS_SIZE
//   → MEM_DM_FIRMWARE_BASE = MEM_LLK_DEBUG_BASE + MEM_LLK_DEBUG_SIZE
//   → [8 DM firmware slots, then 4 TRISC firmware slots]
//   → MEM_TRISC3_FIRMWARE_BASE + MEM_TRISC3_FIRMWARE_SIZE
//   → MEM_DM_GLOBAL_BASE (8 DM cores global storage)
//   → MEM_TRISC_GLOBAL_BASE (4 TRISC cores global storage)
//   → MEM_DM_LOCAL_BASE (8 DM cores local storage)
//   → MEM_TRISC_LOCAL_BASE
//   → MEM_DM_KERNEL_BASE (8 DM kernel code sections)
//   → MEM_NOC_COUNTER_BASE
//   → MEM_FABRIC_COUNTER_BASE (scaled for 8 DMs)
//   → MEM_FABRIC_CONNECTION_LOCK_BASE
//   → MEM_TENSIX_ROUTING_TABLE_BASE
//   → MEM_TENSIX_FABRIC_CONNECTIONS_BASE
//   → MEM_PACKET_HEADER_POOL_BASE
//   → MEM_MAP_END = MEM_PACKET_HEADER_POOL_BASE + MEM_PACKET_HEADER_POOL_SIZE

// Start of available memory region for tile data (LLK transformed address)
// Immediately after all reserved system memory (firmware, DM kernel code, global/local storage,
// counters, routing tables, fabric metadata)
// Physical Boundary: MEM_MAP_END
// Transformed Value: L1_ADDRESS(MEM_MAP_END) = MEM_MAP_END / 16
constexpr std::uint32_t L1_REGION_START = L1_ADDRESS(MEM_MAP_END);

// End of L1 memory (LLK transformed address) - total available L1 size
// This is the absolute upper bound for any L1 address
// Physical Boundary: MEM_L1_BASE + MEM_L1_SIZE
// Transformed Value: L1_ADDRESS(MEM_L1_BASE + MEM_L1_SIZE) = (MEM_L1_BASE + MEM_L1_SIZE) / 16
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
 * The reserved area (MEM_L1_BASE to MEM_MAP_END) contains system firmware, DM kernel code,
 * global/local storage for DM cores and TRISC cores, NoC counters, fabric counters,
 * routing tables, fabric metadata, and packet header pools.
 *
 * QUASAR SPECIFICS:
 * - Significantly larger L1 memory compared to previous architectures
 * - 8 DM cores instead of TRISC-only: Expanded compute architecture
 * - 4 TRISC cores (shared with DM architecture)
 * - Much larger firmware footprint (DM + TRISC firmware)
 * - Larger mailbox: Support for expanded fabric
 * - DM global storage + per-DM local storage
 * - Large DM kernel allocation for DM code
 * - Fabric counter scaled for multiple DMs
 *
 * IMPORTANT: This function takes and compares LLK TRANSFORMED addresses.
 * If you have a physical address, transform it first using L1_ADDRESS():
 *   transformed_addr = L1_ADDRESS(physical_addr) = physical_addr / 16
 * Or pass this function the result from L1_ADDRESS() calls.
 *
 * @param address The LLK-transformed L1 address to validate
 * @return true if address is within valid tile data region [L1_REGION_START, L1_REGION_END)
 */
inline static bool is_valid_L1_address(const std::uint32_t address)
{
    return (address >= ckernel::L1_REGION_START && address < ckernel::L1_REGION_END);
}
