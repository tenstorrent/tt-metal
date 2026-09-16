// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_supported.hpp"

#include <cstdint>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

bool supported_by_codegen(const Tensor& input, uint32_t out_last_dim_elements, const MemoryConfig& output_mem_config) {
    if (input.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        return false;
    }
    // Sharded input or output: the transport has no unshard-to-interleaved hop, so those stay on
    // native.
    if (input.memory_config().is_sharded() || output_mem_config.is_sharded()) {
        return false;
    }
    // Cross-placement calls are out of scope: the reader/writer size their CB slots and NoC
    // transfer bounds from each buffer's own alignment, and the gate below only bounds the plan
    // for the input's own buffer type. A different output buffer type is a different alignment
    // the same plan was never checked against.
    if (output_mem_config.buffer_type() != input.memory_config().buffer_type()) {
        return false;
    }
    // Scalar dtypes only: the transport is a pure byte mover keyed off element_size(), which has
    // no meaning for a block-float shared-exponent format.
    switch (input.dtype()) {
        case tt::tt_metal::DataType::BFLOAT16:
        case tt::tt_metal::DataType::FLOAT32:
        case tt::tt_metal::DataType::INT32:
        case tt::tt_metal::DataType::UINT32:
        case tt::tt_metal::DataType::UINT16: break;
        default: return false;
    }
    if (input.logical_shape().rank() < 1) {
        return false;
    }
    if (out_last_dim_elements == 0) {
        return false;
    }
    if (input.logical_shape().volume() == 0) {
        // Zero-volume reshapes are metadata-only in the free function, before any kernel routing
        // is considered; nothing here needs to serve them.
        return false;
    }
    if (input.storage_type() != ttnn::StorageType::DEVICE) {
        // Not yet on device (e.g. host-side probing); nothing to bound against, and every other
        // condition above answers for a host tensor too.
        return true;
    }

    const uint32_t elem_size = input.element_size();
    const uint32_t old_stick_bytes = static_cast<uint32_t>(input.logical_shape()[-1]) * elem_size;
    const uint32_t new_stick_bytes = out_last_dim_elements * elem_size;
    // out_last_dim_elements is the only output-shape fact this predicate is handed; the transport's
    // plan only needs total output-stick count for its work-unit count, which is the (equal)
    // logical volume divided by the new last dim.
    const uint64_t total_elements = input.logical_shape().volume();
    if (total_elements % out_last_dim_elements != 0) {
        return false;
    }
    const uint32_t num_new_sticks = static_cast<uint32_t>(total_elements / out_last_dim_elements);

    auto* device = input.device();
    const uint32_t input_alignment = device->allocator()->get_alignment(input.memory_config().buffer_type());
    const uint32_t output_alignment = device->allocator()->get_alignment(output_mem_config.buffer_type());
    const uint32_t noc_max_burst_bytes = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    const uint32_t usable_l1_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    const auto plan = ttnn::prim::plan_reshape_rm_arbitrary_transport(
        old_stick_bytes,
        new_stick_bytes,
        num_new_sticks,
        input_alignment,
        output_alignment,
        noc_max_burst_bytes,
        usable_l1_bytes);
    // The factory's CB footprint scales with the stick widths (region_stride, slab_slot_bytes);
    // plan_reshape_rm_arbitrary_transport() reports nabatch == 0 when even the minimum staging
    // area does not fit L1, which the factory would otherwise TT_FATAL on. Reject here so `auto`
    // falls back to native instead of routing to a program that cannot be created.
    return plan.nabatch > 0;
}

bool is_demoted(const Tensor& /*input*/, uint32_t /*out_last_dim_elements*/, const MemoryConfig& /*output_mem_config*/) {
    // No shape is perf-demoted yet; this is the routing extension point once a measured
    // device-time regression is found.
    return false;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
