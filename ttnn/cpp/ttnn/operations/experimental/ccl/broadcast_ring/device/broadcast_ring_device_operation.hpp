// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/broadcast_ring/device/broadcast_ring_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/broadcast_ring/device/broadcast_ring_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/sub_device.hpp>

#include <optional>
#include <variant>

namespace ttnn::prim {

struct BroadcastRingDeviceOperation {
    using operation_attributes_t = BroadcastRingParams;
    using tensor_args_t = BroadcastRingInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<BroadcastRingProgramFactory>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor broadcast_ring(
    const ttnn::Tensor& input_tensor,
    uint32_t sender_ring_index,
    uint32_t cluster_axis,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    tt::tt_fabric::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    uint32_t chunk_size_tiles = 0);

}  // namespace ttnn::prim
