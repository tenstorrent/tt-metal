// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::ccl {

enum class ReceiverL1TestMode : uint8_t { Auto, ForceDirect, ForceReceiver };
enum class ReceiverL1StageMode : uint8_t { Combined, L1Sink, L1Overwrite, DrainOnly };
enum class ReceiverL1NotifyMode : uint8_t { Fused, Split };
enum class ReceiverL1CreditMode : uint8_t { PerSlot, Window, Pipelined };
enum class BankOwnedRunPolicy : uint8_t { Divisor, MaxTail };

// Resolved automatic choices live in AllGatherParams so the program-cache key
// cannot reuse a program compiled for a different receiver protocol or kernel
// schedule.
struct AllGatherReceiverPolicy {
    ReceiverL1TestMode test_mode = ReceiverL1TestMode::Auto;
    ReceiverL1StageMode stage_mode = ReceiverL1StageMode::Combined;
    ReceiverL1NotifyMode notify_mode = ReceiverL1NotifyMode::Fused;
    ReceiverL1CreditMode credit_mode = ReceiverL1CreditMode::Window;
    // Zero selects a bounded automatic group size for pipelined credits.
    uint32_t credit_group_batches = 0;
    bool attribution_enabled = false;
    bool address_attribution_enabled = false;
    bool bank_owned_links = false;
    uint32_t bank_owned_coalesce_mask = 0;
    BankOwnedRunPolicy bank_owned_run_policy = BankOwnedRunPolicy::MaxTail;
    // Bank-interleaved receiver fan-out. Each link owns one
    // receiver core and independent slot ring per DRAM bank assigned to that
    // link, and sender batches rotate across those banks by run index.
    bool interleaved_bank_receivers = false;
    // Zero means dtype-aware automatic selection.
    uint32_t drain_risc_count = 0;
    // Zero means derive the maximum safe value from ordinary-L1 capacity.
    uint32_t slot_count = 0;
    // Zero means the maximum number of rows allowed by the Fabric payload.
    uint32_t batch_rows = 0;
};

// The program-cache hash is computed automatically by reflecting over the members
// below and hashing each one. This is safe only because every field here is a
// stable, structural value.
//
// To add a volatile field later (e.g. a semaphore or raw pointer), do not rely
// on this auto-hashing -- define attribute_names + attribute_values() to list
// exactly what to hash.
struct AllGatherParams {
    int32_t dim = 0;
    MemoryConfig output_mem_config;
    std::optional<uint32_t> cluster_axis;

    // Fabric setup info
    tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED;
    // Per-axis info (an inactive axis has num_devices = 1, num_links = 0, and Linear topology)
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<uint32_t, 2> axis_num_devices{};
    std::array<uint32_t, 2> axis_num_links{};
    uint32_t num_devices = 0;  // number of devices participating in the collective
    size_t packet_size = 0;
    // Host-proved structural eligibility for the native store-and-forward
    // transport. Under Fabric2D every logical edge, including ring wrap, must
    // be one direct physical neighbor hop.
    bool neighbor_unicast_eligible = false;

    // Worker-core selection.
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;

    AllGatherReceiverPolicy receiver_policy;

    // Optional direct-gather geometry. batch_slice_idx selects one dim-0 slab and
    // valid_gather_extent limits the leading per-device extent along dim.
    std::optional<uint32_t> batch_slice_idx;
    std::optional<uint32_t> valid_gather_extent;
};

struct AllGatherInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_tensor;
};

}  // namespace ttnn::operations::ccl
